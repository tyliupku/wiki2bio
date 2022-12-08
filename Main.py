#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:44
# @Author  : Tianyu Liu

import sys
import os
import tensorflow as tf
import time
from SeqUnit import *
from DataLoader import DataLoader
import numpy as np
from PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from datetime import datetime
from preprocess import *
from util import * 


tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 50, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 50, "Number of training epoch.")
tf.app.flags.DEFINE_integer("source_vocab", 20003, 'vocabulary size')
tf.app.flags.DEFINE_integer("field_vocab", 1480, 'vocabulary size')
tf.app.flags.DEFINE_integer("position_vocab", 31, 'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 20003, 'vocabulary size')
tf.app.flags.DEFINE_integer("report", 5000, 'report valid results after some steps')
tf.app.flags.DEFINE_float("learning_rate", 0.0003, 'learning rate')

tf.app.flags.DEFINE_string("mode", 'train', 'train or test')
tf.app.flags.DEFINE_string("load", '0', 'load directory') # BBBBBESTOFAll
tf.app.flags.DEFINE_string("dir", 'processed_data', 'data set directory')
tf.app.flags.DEFINE_integer("limits", 0, 'max data set size')
tf.app.flags.DEFINE_integer("gpu", -1, 'GPU ID for model training')

tf.app.flags.DEFINE_boolean("dual_attention", True, 'dual attention layer or normal attention')
tf.app.flags.DEFINE_boolean("fgate_encoder", True, 'add field gate in encoder lstm')

tf.app.flags.DEFINE_boolean("field", False, 'concat field information to word embedding')
tf.app.flags.DEFINE_boolean("position", False, 'concat position information to word embedding')
tf.app.flags.DEFINE_boolean("encoder_pos", True, 'position information in field-gated encoder')
tf.app.flags.DEFINE_boolean("decoder_pos", True, 'position information in dual attention decoder')


FLAGS = tf.app.flags.FLAGS
last_best = 0.0

gold_path_test = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'processed_data/test/test_split_for_rouge/gold_summary_')
gold_path_valid = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'processed_data/valid/valid_split_for_rouge/gold_summary_')

if FLAGS.load != "0":
    # load an existing model either for further training or testing
    proper_dir = None
    if os.path.isabs(FLAGS.load):
        if 'wiki2bio/results/res' in FLAGS.load:
            proper_dir = FLAGS.load.split('wiki2bio/results/res/')[1].split('/')[0]
            load_dir = FLAGS.load
    else:
        if os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.load)):
            proper_dir = FLAGS.load.strip('.').split('results/res/')[1].split('/')[0]
            load_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.load)
        elif os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/res/' + FLAGS.load)):
            proper_dir = FLAGS.load.split('/')[0]
            load_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/res/' + FLAGS.load)

    assert proper_dir, "Wrong load path -- it must be a subdirectory within the project under the " + \
                       "results/res/ directory with the model name and the loaded epoch, which can be given " + \
                       "either as a model_name/loads/epoch or as an absolute path to the epoch subdirectory " + \
                       "or as a relative path to the epoch subdirectory"

    if FLAGS.mode == 'train':
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/res/' + proper_dir)
        pred_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/evaluation/' + proper_dir)
        save_file_dir = os.path.join(save_dir, 'files')
        log_file = os.path.join(save_dir, 'log_train.txt')
    elif FLAGS.mode == 'test':
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/res/' + FLAGS.load)
        pred_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/evaluation/' + proper_dir)
        log_file = os.path.join(save_dir, 'log_test.txt')
else:
    # train a new model
    prefix = 'model_retrained_by_user_' + datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir =  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/res/' + prefix)
    pred_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/evaluation/' + prefix)
    save_file_dir = os.path.join(save_dir, 'files')
    log_file = os.path.join(save_dir, 'log_train.txt')
    make_dirs()
    try: 
        os.makedirs(pred_dir)
    except OSError:
        if not os.path.isdir(pred_dir):
            raise
    try: 
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise
    try: 
        os.makedirs(pred_dir)
    except OSError:
        if not os.path.isdir(pred_dir):
            raise

pred_path = os.path.join(pred_dir, 'pred_summary_')
pred_beam_path = os.path.join(pred_dir, 'beam_summary_')


def train(sess, dataloader, model):
    global save_dir
    global pred_dir

    if FLAGS.load != "0":
        # Continue training of existing model
        e = 0
        try:
            e = int(FLAGS.load.rstrip('/').split("/")[-1])
        except ValueError:
            pass
        if e > 0:
            # Loaded model from last train epoch
            if e < FLAGS.epoch:
                # Maximum number of epochs not reached yet
                k = e * FLAGS.report
                trainset = dataloader.train_set
                loss, start_time = 0.0, time.time()
                for _ in range(e, FLAGS.epoch):
                    for x in dataloader.batch_iter(trainset, FLAGS.batch_size, True):
                        loss += model(x, sess)
                        k += 1
                        progress_bar(k%FLAGS.report, FLAGS.report)
                        if k % FLAGS.report == 0:
                            cost_time = time.time() - start_time
                            write_log("%d : loss = %.3f, time = %.3f " % (k // FLAGS.report, loss, cost_time))
                            loss, start_time = 0.0, time.time()
                            if k // FLAGS.report >= 1: 
                                ksave_dir = save_model(model, save_dir, k // FLAGS.report)
                                write_log(evaluate(sess, dataloader, model, ksave_dir, 'valid'))
            else:
                # Maximum number of training epochs reached
                print("Model can not be trained further -- maximum number of training epochs reached")
        else:
            # Parsing failed -- can not train model
            print("Model can not be trained further -- last known train epoch missing")
    else:
        # Create and train a new model
        write_log("#######################################################")
        for flag, val in FLAGS.flag_values_dict().iteritems():
            write_log(flag + " = " + str(val))
        write_log("#######################################################")
        trainset = dataloader.train_set
        k = 0
        loss, start_time = 0.0, time.time()
        for _ in range(FLAGS.epoch):
            for x in dataloader.batch_iter(trainset, FLAGS.batch_size, True):
                loss += model(x, sess)
                k += 1
                progress_bar(k%FLAGS.report, FLAGS.report)
                if k % FLAGS.report == 0:
                    cost_time = time.time() - start_time
                    write_log("%d : loss = %.3f, time = %.3f " % (k // FLAGS.report, loss, cost_time))
                    loss, start_time = 0.0, time.time()
                    if k // FLAGS.report >= 1: 
                        ksave_dir = save_model(model, save_dir, k // FLAGS.report)
                        write_log(evaluate(sess, dataloader, model, ksave_dir, 'valid'))



def test(sess, dataloader, model):
    write_log(evaluate(sess, dataloader, model, save_dir, 'test'))

def save_model(model, save_dir, cnt):
    new_dir = os.path.join(save_dir, 'loads')
    nnew_dir = os.path.join(new_dir, str(cnt))
    try: 
        os.makedirs(nnew_dir)
    except OSError:
        if not os.path.isdir(nnew_dir):
            raise
    model.save(nnew_dir)
    return nnew_dir

def evaluate(sess, dataloader, model, ksave_dir, mode='valid'):
    if mode == 'valid':
        # texts_path = "original_data/valid.summary"
        texts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed_data/valid/valid.box.val")
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
    else:
        # texts_path = "original_data/test.summary"
        texts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "processed_data/test/test.box.val")
        gold_path = gold_path_test
        evalset = dataloader.test_set
    
    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []

    k = 0
    for x in dataloader.batch_iter(evalset, FLAGS.batch_size, False):
        predictions, atts = model.generate(x, sess)
        atts = np.squeeze(atts)
        idx = 0
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid == 3:
                        sub = texts[k][np.argmax(atts[tk,: len(texts[k]),idx])]
                        real_sum.append(sub)
                        mask_sum.append("**" + str(sub) + "**")
                    else:
                        real_sum.append(v.id2word(tid))
                        mask_sum.append(v.id2word(tid))
                    unk_sum.append(v.id2word(tid))
                sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                pred_list.append([str(x) for x in real_sum])
                pred_unk.append([str(x) for x in unk_sum])
                pred_mask.append([str(x) for x in mask_sum])
                k += 1
                idx += 1
    write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")


    for tk in range(k):
        with open(gold_path + str(tk), 'r') as g:
            gold_list.append([g.read().strip().split()])

    gold_set = [[gold_path + str(i)] for i in range(k)]
    pred_set = [pred_path + str(i) for i in range(k)]

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_list)
    copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
        (str(F_measure), str(recall), str(precision), str(bleu))

    for tk in range(k):
        with open(pred_path + str(tk), 'w') as sw:
            sw.write(" ".join(pred_unk[tk]) + '\n')

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_unk)
    nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
        (str(F_measure), str(recall), str(precision), str(bleu))

    result = copy_result + nocopy_result 

    print result

    return result



def write_log(s):
    global log_file

    with open(log_file, 'a+') as f:
        f.write(s+'\n')


def main():
    # Stores GPU availability status, if the use of GPU was requested
    use_gpu = False

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    # If the use of GPU was requested in command-line arguments
    if FLAGS.gpu >= 0:
        gpu_id = str(FLAGS.gpu)

        with tf.compat.v1.Session(config=config) as session:
            # Check if the requested GPU ID is available on the system
            for device in session.list_devices():
                dev = device.name.split('/')[-1]
                if dev == 'gpu:' + gpu_id or dev == 'GPU:' + gpu_id:
                    use_gpu = True
                    break

        if not use_gpu:
            print "GPU ID %s not found on the system, the CPU will be used instead" % str(FLAGS.gpu)

    if not use_gpu:
        # Either the requested GPU ID is unavailable or the use of GPU was not requested
        with tf.compat.v1.Session(config=config) as session:
            #copy_file(save_file_dir)
            dataloader = DataLoader(os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.dir), FLAGS.limits)
            model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                            field_size=FLAGS.field_size, pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
                            source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,
                            target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                            field_concat=FLAGS.field, position_concat=FLAGS.position,
                            fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention, decoder_add_pos=FLAGS.decoder_pos,
                            encoder_add_pos=FLAGS.encoder_pos, learning_rate=FLAGS.learning_rate)
            session.run(tf.compat.v1.global_variables_initializer())

            if FLAGS.load != '0':
                model.load(load_dir)
            if FLAGS.mode == 'train':
                train(session, dataloader, model)
            else:
                test(session, dataloader, model)
    else:
        # The requested GPU ID is available on the system
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            with tf.device('/' + dev + ':' + gpu_id):
                #copy_file(save_file_dir)
                dataloader = DataLoader(os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.dir), FLAGS.limits)
                model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                                field_size=FLAGS.field_size, pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
                                source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,
                                target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                                field_concat=FLAGS.field, position_concat=FLAGS.position,
                                fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention, decoder_add_pos=FLAGS.decoder_pos,
                                encoder_add_pos=FLAGS.encoder_pos, learning_rate=FLAGS.learning_rate)
                session.run(tf.compat.v1.global_variables_initializer())

                if FLAGS.load != '0':
                    model.load(load_dir)
                if FLAGS.mode == 'train':
                    train(session, dataloader, model)
                else:
                    test(session, dataloader, model)


if __name__=='__main__':
    main()
