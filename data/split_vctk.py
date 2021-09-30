import os
import shutil
import random

mel_root_dir = '/files/xxx/VC/VCTK/VCTK-Corpus/melspectrogram_vctk_128_all'
audios = os.listdir(mel_root_dir)
classes = sorted(list(set([path.split('_')[0] for path in audios])))
assert len(classes) == 109
classes_audios = {audio.split('_')[0]: [] for audio in classes}
for audio in audios:
    classes_audios[audio.split('_')[0]].append(audio)


def dataset_split():
    os.makedirs(os.path.join(mel_root_dir, 'train'))
    os.makedirs(os.path.join(mel_root_dir, 'test_seen'))
    # os.makedirs(os.path.join(mel_root_dir, 'test_unseen'))

   

    for spk in classes:
        utr_len = len(classes_audios[spk])
        seen_test_len = utr_len // 10
        utrs = classes_audios[spk]
        random.shuffle(utrs)
        seen_utrs = utrs[:seen_test_len]
        train_utrs = utrs[seen_test_len:]
        for utr in seen_utrs:
            shutil.move(os.path.join(mel_root_dir, utr), os.path.join(mel_root_dir, 'test_seen', utr))
        for utr in train_utrs:
            shutil.move(os.path.join(mel_root_dir, utr), os.path.join(mel_root_dir, 'train', utr))

if __name__ == '__main__':
    dataset_split()