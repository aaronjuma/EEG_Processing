import time

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes


def main():

    # Setup
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board_id = BoardIds.UNICORN_BOARD
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    # Basic Calculations
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    time_to_record = sampling_rate*5 # 5 seconds

    # Wait to populate data
    time.sleep(10)
    data = board.get_current_board_data(time_to_record)
    board.stop_stream()
    board.release_session()

    # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    # for demo apply different filters to different channels, in production choose one
    for channel in eeg_channels:
        DataFilter.perform_bandpass(data[channel], BoardShim.get_sampling_rate(board_id), 2.0, 60.0, 1, FilterTypes.BUTTERWORTH, 0)
        DataFilter.perform_lowpass(data[channel], BoardShim.get_sampling_rate(board_id), 60.0, 1, FilterTypes.BUTTERWORTH, 1)
        DataFilter.perform_highpass(data[channel], BoardShim.get_sampling_rate(board_id), 2.0, 1, FilterTypes.BUTTERWORTH, 0)
        # DataFilter.perform_rolling_filter(data[channel], 3, AggOperations.MEAN.value)
        DataFilter.remove_environmental_noise(data[channel], BoardShim.get_sampling_rate(board_id), NoiseTypes.SIXTY.value)

    fig, ax = plt.subplots()
    ax.plot(data[0])
    plt.show()


if __name__ == "__main__":
    main()