import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, AggOperations, WaveletTypes, NoiseEstimationLevelTypes, WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes

class Graph:

    """
    Initializer for the Grapher Class
    Input: config
    Output: None
    """
    def __init__(self):

        # Setting up the board
        # self.board_id = BoardIds.UNICORN_BOARD
        self.board_id = BoardIds.SYNTHETIC_BOARD
        self.board = BoardShim(self.board_id, BrainFlowInputParams())
        self.channels = self.board.get_eeg_channels(self.board_id)

        # Sets up the figure
        self.fig = plt.figure()
        self.ax = self.fig.subplots(len(self.channels))

        # Setting up the window sizes
        self.window_size = 5 # Seconds
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id) # Hz
        self.bin_size = self.window_size*self.sampling_rate

        # Start recording/collecting data
        self.board.prepare_session()
        self.board.start_stream ()

        # Populate the board, will crash if not
        time.sleep(self.window_size)
        start = self.board.get_current_board_data(self.bin_size)
        self.initial_time = start[30,0] #Get the initial timestamp data


    """
    Animate function that will update the graph with new values
    Input: None
    Output: None
    """
    def animate(self, i):

        data = self.board.get_current_board_data(self.bin_size)

        x = data[30] - self.initial_time
        y = data[self.channels[0] : self.channels[-1] + 1]

        #Add Processing Code Here:

        # Updates plot
        for i in self.channels:
            self.ax[i-1].clear()
            self.ax[i-1].set_ylabel(f"Channel {i}")
            DataFilter.detrend(y[i-1], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(y[i-1], self.sampling_rate, 2.0, 60.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            # DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
            #                             FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(y[i-1], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_rolling_filter(y[i-1], 3, AggOperations.MEAN.value)
            DataFilter.perform_rolling_filter(y[i-1], 3, AggOperations.MEDIAN.value)
            DataFilter.perform_wavelet_denoising(y[i-1], WaveletTypes.BIOR3_9, 3,
                                        WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                        WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)
            self.ax[i-1].plot(x, y[i-1], color = 'tab:red')

        self.ax[self.channels[-1]-1].set_xlabel('Time (s)')
        
        # Finishing touch ups on the graph
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title('EEG Real-Time Data')
        plt.tight_layout()


    """
    Multiprocessing Function to Plot the graph in real time
    Input: Multiprocessing Value variables of diameter, speed, and threshold
    Output: None
    """
    def plot(self):

        # Plots the graph in real time
        ani = animation.FuncAnimation(self.fig, self.animate, interval=50, cache_frame_data=False)
        plt.show()

if __name__ == '__main__':
    graph = Graph()
    graph.plot()
