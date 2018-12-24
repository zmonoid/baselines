def atari():
    return dict(
        network='cnn_mlp',
        lr=1e-4,
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        dueling=True,
        hiddens=[256],
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=10000,
        checkpoint_path=None,
    )


def classic_control():
    return dict(
        network='mlp_only',
        hiddens=[64, 64]
    )

def retro():
    return atari()

