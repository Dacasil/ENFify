import unittest
import numpy as np
from enfify import preprocessing


class PreprocessingTestCase(unittest.TestCase):
    def test_downsampling_python(self):
        # Test downsampling_python function
        s_raw = np.random.rand(1000)
        f_s = 44100
        f_ds = 1000
        result = preprocessing.downsampling_python(s_raw, f_s, f_ds)
        self.assertEqual(len(result), int(len(s_raw) * f_ds / f_s))

    def test_list_files_in_directory(self):
        # Test list_files_in_directory function
        input_dir = "/path/to/input/dir"  # TODO: Define input directory
        output_dir = "/path/to/output/dir"
        expected_result = ...  # TODO: Define expected result
        result = preprocessing.list_files_in_directory(input_dir, output_dir)
        self.assertEqual(result, expected_result)

    def test_downsampling(self):
        # Test downsampling function
        sig = np.random.rand(1000)
        fs = 44100
        fs_down = 1000
        result = preprocessing.downsampling(sig, fs, fs_down)
        self.assertEqual(len(result), int(len(sig) * fs_down / fs))

    def test_bandpass_filter(self):
        # Test bandpass_filter function
        sig = np.random.rand(1000)
        lowcut = 50
        highcut = 2000
        fs = 44100
        order = 4
        result = preprocessing.bandpass_filter(sig, lowcut, highcut, fs, order)
        self.assertEqual(len(result), len(sig))

    def test_generate_random_number(self):
        # Test generate_random_number function
        min_value = 0
        max_value = 1
        decimal_places = 2
        result = preprocessing.generate_random_number(min_value, max_value, decimal_places)
        self.assertTrue(min_value <= result <= max_value)

    def test_generate_s_tone(self):
        # Test generate_s_tone function
        Fs = 44100
        f0 = 50
        phi0 = 0
        M = 1000
        result = preprocessing.generate_s_tone(Fs, f0, phi0, M)
        self.assertEqual(len(result), M)

    def test_create_tones(self):
        # Test create_tones function
        Fs = 44100
        M = 1000
        toneSample_num = 10
        result = preprocessing.create_tones(Fs, M, toneSample_num)
        self.assertEqual(len(result), toneSample_num)

    def test_random_signal(self):
        # Test random_signal function
        AMPLITUDE = 1
        DURATION = 1
        F_DS = 1000
        NOMINAL_ENF = 50
        PM_NOMINAL = 0.1
        CUT_SAMPLES_LIMIT = 100
        result = preprocessing.random_signal(
            AMPLITUDE, DURATION, F_DS, NOMINAL_ENF, PM_NOMINAL, CUT_SAMPLES_LIMIT
        )
        self.assertEqual(len(result), int(DURATION * F_DS))

    def test_cut_tones(self):
        # Test cut_tones function
        sig = np.random.rand(1000)
        F_DS = 1000
        result = preprocessing.cut_tones(sig, F_DS)
        self.assertEqual(len(result), len(sig))


if __name__ == "__main__":
    unittest.main()
