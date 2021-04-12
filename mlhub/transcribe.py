import utils
import argparse
import os
import sys

from mlhub.pkg import is_url
from mlhub.utils import get_cmd_cwd
import requests


def main():

    # -----------------------------------------------------------------------
    # Load pre-built models
    # -----------------------------------------------------------------------

    scorer = os.path.join(os.getcwd(), "deepspeech-0.9.3-models.scorer")
    model = os.path.join(os.getcwd(), "deepspeech-0.9.3-models.pbmm")

    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')

    parser.add_argument('--model', default= model,
                         help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', default=scorer,
                        help='Path to the external scorer file')
    parser.add_argument('--audio', required=True,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--beam_width', type=int,
                        help='Beam width for the CTC decoder')
    parser.add_argument('--lm_alpha', type=float,
                        help='Language model weight (lm_alpha). If not specified, use default from the scorer package.')
    parser.add_argument('--lm_beta', type=float,
                        help='Word insertion bonus (lm_beta). If not specified, use default from the scorer package.')
    parser.add_argument('--extended', required=False, action='store_true',
                        help='Output string from extended metadata')
    parser.add_argument('--json', required=False, action='store_true',
                        help='Output json from metadata with timestamp of each word')
    parser.add_argument('--candidate_transcripts', type=int, default=3,
                        help='Number of candidate transcripts to include in JSON output')
    parser.add_argument('--hot_words', type=str,
                        help='Hot-words and their boosts.')
    args = parser.parse_args()

    path = args.audio

    if is_url(path):
        response = requests.get(path)
        if response.status_code != 200:
            print(f"The URL does not appear to exist. Please check.\n    {path}")
            sys.exit()
    else:
        path = os.path.join(get_cmd_cwd(), path)
        print(get_cmd_cwd())
    print(path)
    utils.deepspeech(args.model, args.scorer, path, "transcribe", False, args.beam_width, args.lm_alpha,
                     args.lm_beta, args.extended, args.json, args.candidate_transcripts, args.hot_words)


if __name__ == '__main__':
    main()
