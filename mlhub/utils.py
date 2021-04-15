#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import tempfile

import numpy as np
import shlex
import subprocess
import sys
import wave
import json
import os
from urllib.parse import urlparse
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from deepspeech import Model
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(
        quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno,
                      'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


def metadata_to_string(metadata):
    return ''.join(token.text for token in metadata.tokens)


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def metadata_json_output(metadata):
    json_result = dict()
    json_result["transcripts"] = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    } for transcript in metadata.transcripts]
    return json.dumps(json_result, indent=2)


def load(model, scorer, verbose=True, beam_width="", lm_alpha="", lm_beta="", hot_words=""):
    """ Load models"""

    model_load_start = timer()
    # sphinx-doc: python_ref_model_start
    ds = Model(model)
    # sphinx-doc: python_ref_model_stop
    model_load_end = timer() - model_load_start
    if verbose==True:
        print('\nLoading model from files {}'.format(model), file=sys.stderr)
        print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    if beam_width:
        ds.setBeamWidth(beam_width)

    desired_sample_rate = ds.sampleRate()

    if scorer:
        if verbose == True:
            print('Loading scorer from files {}'.format(scorer), file=sys.stderr)
        scorer_load_start = timer()
        ds.enableExternalScorer(scorer)
        scorer_load_end = timer() - scorer_load_start
        if verbose == True:
            print('Loaded scorer in {:.3}s.'.format(scorer_load_end), file=sys.stderr)

        if lm_alpha and lm_beta:
            ds.setScorerAlphaBeta(lm_alpha, lm_beta)

    if hot_words:
        if verbose == True:
            print('Adding hot-words', file=sys.stderr)
        for word_boost in hot_words.split(','):
            word, boost = word_boost.split(':')
            ds.addHotWord(word, float(boost))
    return ds, desired_sample_rate


def deepspeech(ds, desired_sample_rate, audio, type, verbose=True, extended="", json="",
               candidate_transcripts=""):
    """ Run deepspeech"""

    # Check if the audio is a file or url
    result = urlparse(audio)

    if all([result.scheme, result.netloc, result.path]):
        if audio.find('/'):
            audio_name = audio.rsplit('/', 1)[1]
        else:
            audio_name = "download.wav"

        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(audio, stream=True, headers=headers)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdirname:
            audio = os.path.join(tmpdirname, audio_name)
            open(audio, 'wb').write(r.content)
            fin = wave.open(audio, 'rb')
            fs_orig = fin.getframerate()
            if fs_orig != desired_sample_rate:
                if verbose == True:
                    print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic '
                          'speech recognition.'.format(
                        fs_orig, desired_sample_rate), file=sys.stderr)
                fs_new, audio = convert_samplerate(audio, desired_sample_rate)
            else:
                audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

            audio_length = fin.getnframes() * (1 / fs_orig)
            fin.close()

            if verbose == True:
                print('\nRunning inference to transcribe the audio...', file=sys.stderr)
            inference_start = timer()
            # sphinx-doc: python_ref_inference_start
            if extended:
                print("\n\t" + metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0]) + "\n")
            elif json:
                print("\n\t" + metadata_json_output(ds.sttWithMetadata(audio, candidate_transcripts)) + "\n")
            else:
                print("\n\t" + ds.stt(audio) + "\n")
            # sphinx-doc: python_ref_inference_stop
            inference_end = timer() - inference_start
            if verbose == True:
                print('\nInference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length) + "\n",
                      file=sys.stderr)

    else:
        fin = wave.open(audio, 'rb')
        fs_orig = fin.getframerate()
        if fs_orig != desired_sample_rate:
            if verbose == True:
                print(
                    'Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(
                        fs_orig, desired_sample_rate), file=sys.stderr)
            fs_new, audio = convert_samplerate(audio, desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

        audio_length = fin.getnframes() * (1 / fs_orig)
        fin.close()

        if verbose == True:
            print('Running inference to transcribe the audio...', file=sys.stderr)
        inference_start = timer()
        # sphinx-doc: python_ref_inference_start
        if type == "demo":
            if extended:
                print("\n\t" + metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0]))
            elif json:
                print("\n\t" + metadata_json_output(ds.sttWithMetadata(audio, candidate_transcripts)))
            else:
                print("\n\t\"" + ds.stt(audio) + "\"")
                print("\n\t" + ds.stt(audio))
        elif type == "transcribe":
            if extended:
                print(metadata_to_string(ds.sttWithMetadata(audio, 1).transcripts[0]))
            elif json:
                print(metadata_json_output(ds.sttWithMetadata(audio, candidate_transcripts)))
            else:
                print(ds.stt(audio))

        # sphinx-doc: python_ref_inference_stop
        inference_end = timer() - inference_start
        if verbose == True:
            print('\nInference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length) + "\n",
                  file=sys.stderr)
