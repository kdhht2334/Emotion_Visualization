import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import gradio as gr
import codecs
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset
from img2vec_pytorch import Img2Vec

import matplotlib.pyplot as plt

import cv2


class FER2013Dataset(Dataset):
    """Face expression recognition database."""

    def __init__(self, file_path, size_change):
        """
        Args:
            file_path (string): Path to the csv file with emotion, pixel, and usage.
        """
        self.file_path = file_path
        self.size_change = size_change

        # TODO : fixed according to the emotion label
        self.classes = (
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        )

        with open(self.file_path) as f:
            self.total_images = len(f.readlines()) - 1  # reduce 1 for row of column

    def __len__(self):  # to return total images when call `len(dataset)`
        return self.total_images

    def __getitem__(self, idx):  # to return image and emotion when call `dataset[idx]`
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.file_path) as f:
            emotion, img = f.readlines()[idx + 1].split(",")

        emotion = int(emotion)
        img = img.replace('"', "")
        img = img.split(" ")
        img = np.array(img, "int")
        img = img.reshape(48, 48).astype("float32")
        if self.size_change:
            img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        return {"image": img, "emotion": emotion}


def preprocess_data():
    fer_path = "train.csv"
    size_change = -1

    dataset = FER2013Dataset(fer_path, size_change)
    img0vec = Img2Vec(cuda=True)

    db_list, label_list = [], []
    for i in range(100):
        aaa = np.expand_dims(dataset[i]["image"], axis=-3)
        db = np.concatenate((aaa, aaa, aaa), axis=-3)
        db_list.append(Image.fromarray(db.astype("uint8"), "RGB"))

        label_list.append(dataset[i]["emotion"])

    vec = img0vec.get_vec(db_list, tensor=True).squeeze().detach().cpu().numpy()

    reduce = umap.UMAP(n_components=3)
    embedding = reduce.fit_transform(vec)
    target = np.asarray(label_list)

    df = pd.DataFrame(embedding, columns=("x", "y", "z"))
    df["class"] = target
    df["size"] = 0.5
    df["showgrid"] = False

    labels = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral",
    }
    df["class"].replace(labels, inplace=True)
    return df


def plot_data(df):
    name = "FER2013 visualization"
    camera = dict(
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.1, y=0.1, z=1)
    )

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        size="size",
        color="class",
        symbol="class",
        title="FER2013 visualization",
    )
    fig.update_layout(scene_camera=camera, title=name)
    fig.update_layout(paper_bgcolor="black")
    fig.update_traces(marker_size=1.2)
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="black",
                showbackground=False,
                showticklabels=False,
                zerolinecolor="black",
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="black",
                showbackground=False,
                showticklabels=False,
                zerolinecolor="black",
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="black",
                showbackground=False,
                showticklabels=False,
                zerolinecolor="black",
            ),
        ),
    )
    fig.show()
    fig.write_html("test1.html")
    f = codecs.open("test.html", "r", "utf=-8")
    doc = BeautifulSoup(f)
    return str(doc)


def main():
    title = "FER 2013 visualization"
    description = """_summary_: None
    """
    df = preprocess_data()

    with gr.Blocks() as app:
        with gr.Accordion("Open details about VISUALIZATION project"):
            gr.Markdown("Look at me...")

        gr.HTML(f"{title}")
        gr.Markdown(description)

        btn = gr.Button("Visualize").style(full_width=True)
        btn.click(
            plot_data,
            inputs=[gr.DataFrame(df, max_rows=5)],
            outputs=[gr.outputs.HTML()],
        )

    app.queue().launch()


if __name__ == "__main__":
    main()
