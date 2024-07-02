import time
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import atexit

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets

class EEGController:

    # Initializer
    def __init__(self):

        # Setting up the board
        params = BrainFlowInputParams()
        params.serial_number = 'UN-2023.02.30'
        self.board_id = BoardIds.UNICORN_BOARD
        self.board = BoardShim(self.board_id, params)

        # Getting specific board details
        self.channels = self.board.get_eeg_channels(self.board_id) #EEG Channels
        self.timestamp_channel = self.board.get_timestamp_channel(self.board_id) # Timestamp channel
        self.marker_channel = self.board.get_marker_channel(self.board_id) # Marker channel for synchronization
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id) # Hz

        # Start recording/collecting data
        self.board.prepare_session()
        self.board.start_stream ()

        # Populate the board, will crash if not
        time.sleep(0.2)
        start = self.board.get_current_board_data(200)

        # Get timestamp data
        self.initial_time = start[self.timestamp_channel,0] #Get the initial timestamp data

        # Reseting Marker
        self.marker_id = 0

        # If PyshcoPy experiment ends prematurely, safely close this controller
        atexit.register(self.close)

    # Return timestamp data
    def setMarker(self):
        self.marker_id = self.marker_id + 1
        self.board.insert_marker(self.marker_id)
        return self.marker_id
    

    def close(self):
        # Get EEG data from board and stops EEG session
        data = self.board.get_board_data()
        self.board.stop_stream()
        self.board.release_session()

        # Get x and y data
        data[self.timestamp_channel] = data[self.timestamp_channel] - self.initial_time
        # new_data = np.array(x)
        # y = data[self.channels[0] : self.channels[-1] + 1]
        # new_data = np.vstack((x, y))
        # new_data = np.vstack((new_data, data[self.marker_channel]))

        np.savetxt("eeg_data.csv", np.transpose(data), delimiter=",")

