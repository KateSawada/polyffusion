from . import AttrDict

chd_z_dim = 256
n_bars = 1
beat_resolution = 4
beats_per_bar = 4

params = AttrDict(
    # Training params
    batch_size=16,
    max_epoch=40,
    learning_rate=1e-3,
    max_grad_norm=10,
    fp16=True,

    # Data params
    num_workers=4,
    pin_memory=True,

    weights=[1, 0.5],  # constant scheduler
    beta=0.1,
    tfr=[(0.6, 0), (0.5, 0), (0.5, 0)],  # teacher forcing rate

    n_bars=n_bars,
    bar_resolution=beat_resolution * beats_per_bar,

    chd_n_step=n_bars * beats_per_bar,
    chd_input_dim=36,
    chd_z_input_dim=512,
    chd_hidden_dim=512,
    chd_z_dim=chd_z_dim,

    txt_emb_size=256,
    txt_hidden_dim=1024,
    txt_z_dim=256,
    txt_num_channel=10,

    pno_dec_dur_hid_size=64,
    pno_z_size=512,
)
