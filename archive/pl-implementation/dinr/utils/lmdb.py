import io
import logging
import os
import pickle

import lmdb
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class LmdbDataset(Dataset):
    def __init__(self, path, max_readers=1, transform=None, max_images=None):
        super(LmdbDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.max_readers = max_readers
        self.max_images = max_images

        env = self.init_env()
        with env.begin(write=False) as txn:
            cache_path = os.path.join(path, "cached_keys.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as fp:
                    self.keys = pickle.load(fp)
            else:
                self.keys = [key for key, _ in txn.cursor()]
                with open(cache_path, 'wb') as fp:
                    pickle.dump(self.keys, fp)
            assert len(self.keys) == txn.stat()['entries']

    def init_env(self):
        return lmdb.open(self.path, max_readers=self.max_readers, readonly=True, lock=False, readahead=False,
                         meminit=False)

    def __getitem__(self, index):
        if not hasattr(self, 'env'):
            self.env = self.init_env()

        with self.env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # using opencv
        # img_buffer = np.frombuffer(imgbuf, dtype=np.uint8)
        # img = cv2.imdecode(img_buffer, 1)[::-1]  # BGR to RGB
        # img = torch.as_tensor(np.ascontiguousarray(np.ascontiguousarray(img.transpose(2, 0, 1))))

        return {"img": img}

    def __len__(self):
        if self.max_images is not None:
            return min(len(self.keys), self.max_images)
        return len(self.keys)
