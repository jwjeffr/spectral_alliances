import requests
from io import BytesIO

import matplotlib.pyplot as plt
import networkx as nx
import flagpy
from PIL import Image, ImageOps

from data_collection import Data


def main():

    data = Data()

    G = nx.Graph()
    misnamed_countries = {
        "Comoros",
        "Czech Republic",
        "Gambia",
        "Maldives",
        "Netherlands",
        "Philippines",
        "United Arab Emirates",
        "United Kingdom",
        "United States",
    }
    for country in data.unique_countries:

        if country == "Taiwan":
            r = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Flag_of_the_Republic_of_China.svg/1920px-Flag_of_the_Republic_of_China.svg.png")
            flag = Image.open(BytesIO(r.content))

        elif country in misnamed_countries:
            flag = flagpy.get_flag_img(f"The {country}")

        elif country == "Republic of Ireland":
            flag = flagpy.get_flag_img("Ireland")

        else:
            flag = flagpy.get_flag_img(country)

        flag = ImageOps.expand(flag.resize((130, 90)), border=(20, 10, 20, 10), fill="black")

        G.add_node(country, image=flag)

    adjacency_matrix = data.get_adjacency_matrix()

    for i in range(data.num_countries):
        for j in range(data.num_countries):
            if adjacency_matrix[i, j] == 0.0 or i == j:
                continue
            country_one, country_two = data.index_to_country[i], data.index_to_country[j]
            G.add_edge(country_one, country_two, weight=adjacency_matrix[i, j])

    pos = nx.kamada_kawai_layout(G, scale=0.5)
    fig, ax = plt.subplots()

    nx.draw_networkx_edges(G, pos=pos, ax=ax, width=0.1)

    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
    icon_center = icon_size / 2.0

    for n in G.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        a.imshow(G.nodes[n]["image"])
        a.axis("off")

    fig.savefig("relations_graph.png", dpi=800)


if __name__ == "__main__":

    main()
