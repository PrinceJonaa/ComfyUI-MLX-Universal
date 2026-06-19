import unittest
from diffusionkit.utils import bytes2gigabytes

class TestUtils(unittest.TestCase):
    def test_bytes2gigabytes(self):
        # 0 bytes should be 0 gigabytes
        self.assertEqual(bytes2gigabytes(0), 0)

        # 1 GB in bytes
        one_gb_in_bytes = 1024**3
        self.assertEqual(bytes2gigabytes(one_gb_in_bytes), 1.0)

        # 2.5 GB in bytes
        two_and_half_gb = int(2.5 * 1024**3)
        self.assertEqual(bytes2gigabytes(two_and_half_gb), 2.5)

if __name__ == '__main__':
    unittest.main()
