import tensorflow as tf

def build_mlp(input_dim, hidden=(128,64), n_classes=3):
    tf.random.set_seed(42)
    inp = tf.keras.Input(shape=(input_dim,), name="x")
    x = inp
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
 
