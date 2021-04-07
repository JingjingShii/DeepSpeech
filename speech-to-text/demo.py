import utils
import argparse
import os
import tarfile
from mlhub.pkg import mlask, mlcat

def main():
    parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
    # parser.add_argument('--model', required=True,
    #                      help='Path to the model (protocol buffer binary file)')
    # parser.add_argument('--scorer', required=False,
    #                     help='Path to the external scorer file')
    # parser.add_argument('--audio', required=True,
    #                     help='Path to the audio file to run (WAV format)')
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

    scorer = os.path.join(os.getcwd(), "cache/resources/deepspeech-0.9.3-models.scorer")
    model = os.path.join(os.getcwd(), "cache/resources/deepspeech-0.9.3-models.pbmm")
    audio = os.path.join(os.getcwd(), "cache/resources/audio-0.9.3.tar.gz")

    tar = tarfile.open(audio, "r:gz")
    tar.extractall()
    tar.close()

    audio_path = os.path.join(os.getcwd(), "audio")

    audio_file_list = []

    for filename in os.listdir(audio_path):
        if not filename.startswith(".") and filename.endswith("wav"):
            audio_file_list.append(os.path.join(os.getcwd(), "audio/"+filename))

    mlcat("Deepspeech", "Welcome to a demo of pre-built model for deepspeech. TThis model is trained by machine learning techniques based "
                        "on Baidu's Deep Speech research paper (https://arxiv.org/abs/1412.5567), and implemented by Mozilla."
                        "In this speech recognition system, the audio will be converted into text.\n")
    mlask()
    mlcat("Demonstration", "Next we will demonstrate three examples from Mozilla example audio files.\n")
    mlask()
    mlcat("First","The transcription in the first audio is Experience proves this")
    mlask()
    utils.deepspeech(model, scorer, audio_file_list[0], args.beam_width, args.lm_alpha,
                     args.lm_beta, args.extended, args.json, args.candidate_transcripts, args.hot_words)
    mlask()
    mlcat("Second", "The transcription in the second audio is Why should one halt on the way?")
    mlask()
    utils.deepspeech(model, scorer, audio_file_list[1], args.beam_width, args.lm_alpha,
                     args.lm_beta, args.extended, args.json, args.candidate_transcripts, args.hot_words)
    mlask()
    mlcat("Third", "The transcription in the third audio is Your power is sufficient I said")
    mlask()
    utils.deepspeech(model, scorer, audio_file_list[2], args.beam_width, args.lm_alpha,
                     args.lm_beta, args.extended, args.json, args.candidate_transcripts, args.hot_words)


if __name__ == '__main__':
    main()
