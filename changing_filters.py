def build_keras_cnn(input_shape, num_classes):
    f1, f2 = [int(x * WIDTH_MULT) for x in FILTERS_SIMPLE]

    model = Sequential()
    model.add(Conv2D(filters=f1, kernel_size=(5, 5), padding='same',
                     activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=f2, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
