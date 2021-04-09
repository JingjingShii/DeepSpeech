# deepspeech 
This [MLHUB](https://mlhub.ai/) provides a demonstration and command line tools built from the [pre-trained deepspeech model](https://deepspeech.readthedocs.io/en/v0.9.3/) provided by Mozilla. The model is trained by machine learning techniques based on [Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). This package can take an audio file as the input and transcribe it to the text as the output. 

The deepspeech source code is available from [https://github.com/mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech).

Note: this pre-trained model only supports English. 

## Usage

* To install mlhub (Ubuntu 20.04 LTS)
  ```console
  $ pip3 install mlhub
  $ ml configure
  ```

* To install and configure the demo
  ```console
  $ ml install JingjingShii/deepspeech
  $ ml configure deepspeech
  $ ml readme deepspeech
  $ ml commands deepspeech
  ```
* Run demo
  ```console
  $ ml demo deepspeech
  ```

* Command line tools
  ```console
  $ ml transcribe deepspeech [(--model) <pre-trained model>] [(--scorer) <external score file>] [(--audio) <audio file>]
  [(--verbose) <if print out all the message>] [(--beam_width) <beam width for the CTC decoder>] [(--lm_alpha) <langauge model weight>] 
  [(--lm_beta) <word insertion bonus>] [(--version) <print version and exists>] [(--extended) <Output string from extended metadata>] 
  [(--json) <Output json from metadata with timestamp of each word>] [(--candidate_transcripts) <number of candidate transcripts to include in JSON output>]
  [(--hot_words) <hot-words>]
  ```
* Re-sample your own audio file

  Since the pre-trained model only supports 16kHz audio file, here we use```sox```to resample the audio file
  ```console
  $ sudo apt-get install -y sox
  $ sox original.wav -r 16000 after.wav
  ```
