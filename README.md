# speech_inpainting

This repository contains code to inpaint missing parts of a speech audio signal using advanced deep learning techniques published on INTERSPEECH 2023 under the title "Sequence-to-Sequence Multi-Modal Speech In-Painting".

Speech in-painting is the task of regenerating missing audio contents using reliable context information. Despite various recent studies in multi-modal perception of audio in-painting, there is still a need for an effective infusion of visual and auditory information in speech in-painting. In this paper, we introduce a novel sequence-to-sequence model that leverages the visual information to in-paint audio signals via an encoder-decoder architecture. The encoder plays the role of a lip-reader for facial recordings and the decoder takes both encoder outputs as well as the distorted audio spectrograms to restore the original speech. Our model outperforms an audio-only speech in-painting model and has comparable results with a recent multi-modal speech in-painter in terms of speech quality and intelligibility metrics for distortions of 300 ms to 1500 ms duration, which proves the effectiveness of the introduced multi-modality in speech in-painting.

![Drawing2-3](https://github.com/MahsaElyaderani/speech_inpainting/assets/90406947/e64fa521-3afc-4635-b37e-e11edf49459e=100x100)

