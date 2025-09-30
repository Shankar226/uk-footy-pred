import tensorflow as tf

def build_team_embed_model(team_vocab, tab_dim, n_classes=3, emb_dim=16):
    tf.random.set_seed(42)
    x_tab = tf.keras.Input(shape=(tab_dim,), name="x_tab")
    hteam = tf.keras.Input(shape=(), dtype=tf.string, name="home")
    ateam = tf.keras.Input(shape=(), dtype=tf.string, name="away")

    team_lookup = tf.keras.layers.StringLookup(vocabulary=team_vocab, mask_token=None)
    team_emb = tf.keras.layers.Embedding(len(team_vocab)+1, emb_dim)

    x = tf.keras.layers.Dense(128, activation="relu")(x_tab)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    x = tf.concat([x,
                   team_emb(team_lookup(hteam)),
                   team_emb(team_lookup(ateam))], axis=1)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model([x_tab, hteam, ateam], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
 
