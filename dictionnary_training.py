
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import os

def nmf_l1_train(examples, n_latent_variables, n_iter=200, gamma=None, W=None, H=None):
    '''
    NMF using the Itakura-Saito divergence and l1 regularization for sparse activations.

    Parameters
    ----------
    examples: list of array_like (n_examples, n_bins, n_frames)
        Tensor containing all the spectrograms to decompose
    n_latent_variables: int
        Number of latent variables in the NMF
    n_iter: int
        Number of iterations
    gamma: float
        Regularization paramters
    W: array_like, optional
        Initialization for W
    H: array_like, optional
        Initialization for activations
    '''

    # first concatenate all examples spectrogram into one matrix
    if isinstance(examples, np.ndarray):
        examples = [e for e in examples]
    V = np.concatenate(examples, axis=1)
    V /= V.max()
    V += 1e-10  # because of zero frames

    n_bins, n_frames = V.shape
        
    # initialize if necessary
    if W is None:
        pwr_psd = np.mean(V, axis=1)  # average spectral power
        W = (0.1 + np.abs(np.random.randn(n_bins, n_latent_variables))) * pwr_psd[:,None]
    if H is None:
        pwr_act = np.mean(V, axis=0)  # average activation power
        H = (0.1 + np.abs(np.random.randn(n_latent_variables, n_frames))) * pwr_act[None,:]

    C0 = V / np.dot(W,H)
    objective_value = np.sum(C0 - np.log(C0) - 1) + np.sum(H)

    for epoch in range(n_iter):

        # recompute approximation
        V_hat = np.dot(W,H)

        # convenient quantities
        C2 = 1 / V_hat
        C0 = V * C2     # V / V_hat
        C1 = C0 * C2    # V / V_hat ** 2

        # track progress
        new_objective_value = np.sum(C0 - np.log(C0) - 1) + np.sum(H)
        progress = new_objective_value - objective_value
        objective_value = new_objective_value
        print(epoch,'-- Objective value:', objective_value, 'progress:', progress)

        # Update H
        H *= np.sqrt(np.dot(W.T, C1) / (np.dot(W.T, C2) + gamma))

        # Update W
        W *= np.sqrt(np.dot(C1, H.T) / np.dot(C2, H.T))

    return W, H


if __name__ == '__main__':

    stft_win_len = 2048
    n_speakers = 25  # number of speakers per gender
    n_latent_variables = 500

    # add an environment variable with the TIMIT location
    # e.g. /path/to/timit/TIMIT
    try:
        timit_path = os.environ['TIMIT_PATH']
    except:
        raise ValueError('An environment variable ''TIMIT_PATH'' pointing to the TIMIT base location is needed.')

    # Load the corpus, be patient
    print('Load TIMIT corpus...')
    corpus = pra.TimitCorpus(timit_path)
    corpus.build_corpus()

    # let's find all the sentences from male speakers in the training set
    male_speakers_test = list(set([s.speaker for s in filter(lambda x: x.sex == 'M', corpus.sentence_corpus['TEST'])]))
    male_speakers_train = list(set([s.speaker for s in filter(lambda x: x.sex == 'M', corpus.sentence_corpus['TRAIN'])]))
    female_speakers_test = list(set([s.speaker for s in filter(lambda x: x.sex == 'F', corpus.sentence_corpus['TEST'])]))
    female_speakers_train = list(set([s.speaker for s in filter(lambda x: x.sex == 'F', corpus.sentence_corpus['TRAIN'])]))

    print('Pick a subset of', n_speakers, 'speakers')
    training_set_speakers = male_speakers_train[:n_speakers] + female_speakers_train[:n_speakers]
    training_set_sentences = filter(lambda x: x.speaker in training_set_speakers, corpus.sentence_corpus['TRAIN'])

    # compute all the spectrograms
    print('Compute all the spectrograms')
    window = np.sqrt(pra.cosine(stft_win_len))  # use sqrt because of synthesis
    # X is (n_sentences, n_channel, n_frame)
    X = [pra.stft(sentence.samples, stft_win_len, stft_win_len // 2, win=window, transform=np.fft.rfft).T for sentence in training_set_sentences]
    examples = [np.abs(spectrogram)**2 for spectrogram in X]

    print('Train the dictionary...')

    W, H = nmf_l1_train(examples, n_latent_variables, n_iter=200, gamma=0.1)

