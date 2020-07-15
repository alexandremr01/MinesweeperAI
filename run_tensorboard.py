from tensorboard import program

# Tensorboard is a useful tool for accompanying the training of your model.
# Run this script, then open the url in your browser.
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'logs'])
tb.launch()
print('Tensorboard is running. Please, connect your browser to: http://localhost:6006.')

while True:
    continue
