'''
Potential areas i can change (number of filters)
'''

# First convolutional block
model.add(Conv2D(filters=8,  kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
# ...
# Second convolutional block
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))


Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
Conv2D(64, (3, 3), activation='relu'),
Conv2D(96, (3, 3), activation='relu'),
Conv2D(128, (3, 3), activation='relu'),

