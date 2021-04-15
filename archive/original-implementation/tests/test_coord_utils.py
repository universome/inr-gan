import sys; sys.path.append('.')

from src.dataloaders.imagenet_vs import aspect_ratio_to_wh


def test_aspect_ratio_to_wh():
    for w in range(16, 1024):
        for h in range(16, 1024):
            ar = w / h
            w_rec, h_rec = aspect_ratio_to_wh(ar, max(w, h))
            assert w == w_rec, f"Wrong w_rec for w: {w}, h: {h} --- {w_rec, h_rec}"
            assert h == h_rec, f"Wrong h_rec for w: {w}, h: {h} --- {w_rec, h_rec}"
