"""Execute Train_JulesVerne_RNN.ipynb locally, writing into models_v2/."""
import sys
import nbformat
from nbclient import NotebookClient

nb = nbformat.read("Train_JulesVerne_RNN.ipynb", as_version=4)
for c in nb.cells:
    if c.cell_type == "code":
        c.source = c.source.replace("Path('models/rnn-split.keras')", "Path('models_v2/rnn-split.keras')")
        c.source = c.source.replace("Path(f'models/{name}.json')", "Path(f'models_v2/{name}.json')")
        c.source = c.source.replace("verbose=1)", "verbose=1)\nclass Log(keras.callbacks.Callback):\n    def on_epoch_end(self, e, logs=None):\n        print(f'epoch {e+1}: {logs}', file=sys.__stderr__, flush=True)")
        c.source = c.source.replace("callbacks=[checkpoint]", "callbacks=[checkpoint, Log()], verbose=2")
client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}})
try:
    client.execute()
finally:
    nbformat.write(nb, "models_v2/Train_JulesVerne_RNN.executed.ipynb")
    print("saved executed notebook", file=sys.stderr)
