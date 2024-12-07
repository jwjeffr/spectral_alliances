from itertools import combinations

import folium
import numpy as np
import matplotlib.pyplot as plt

from clustering import spectral_clustering
from data_collection import Data


def main():

    data = Data()

    pca_vectors, categories = spectral_clustering(data.get_adjacency_matrix(), num_categories=3)

    country_to_category = {
        country: int(categories[data.country_to_index[country]]) for country in data.unique_countries
    }
    colors = {0: "#ff0000", 1: "#009bff", 2: "#ff8300"}
    markers = {0: "o", 1: "^", 2: "s"}

    def style_function(feature):

        try:
            category = country_to_category[feature["properties"]["ADMIN"]]
            fill_color = colors[category]
        except KeyError:
            fill_color = "white"

        return {"fillColor": fill_color, "color": "black", "fillOpacity": 0.7, "weight": 2}

    m = folium.Map(location=(30, 10), zoom_start=3)
    folium.GeoJson(
        "map.geojson",
        style_function=style_function
    ).add_to(m)
    m.save("map.html")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for x in np.unique(categories):
        ax.scatter(*pca_vectors[categories == x].T, color=colors[x], marker=markers[x], alpha=0.8)
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.set_zlabel("principal component 3")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    min_, max_ = np.min(pca_vectors, axis=0), np.max(pca_vectors, axis=0)
    ax.set_xlim((min_[0], max_[0]))
    ax.set_ylim((min_[1], max_[1]))
    ax.set_zlim((min_[2], max_[2]))

    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    hungary_vector = pca_vectors[data.country_to_index["Hungary"], :]
    austria_vector = pca_vectors[data.country_to_index["Austria"], :]
    slovakia_vector = pca_vectors[data.country_to_index["Slovakia"], :]

    assert np.all(np.isclose(hungary_vector, austria_vector)) and np.all(np.isclose(austria_vector, slovakia_vector))

    x = 0.03
    vertices = [
        hungary_vector + np.array([x, x, x]),
        hungary_vector - np.array([x, x, x]),
        hungary_vector + np.array([-x, x, x]),
        hungary_vector - np.array([-x, x, x]),
        hungary_vector + np.array([x, -x, x]),
        hungary_vector - np.array([x, -x, x]),
        hungary_vector + np.array([x, x, -x]),
        hungary_vector - np.array([x, x, -x])
    ]

    for v1, v2 in combinations(vertices, 2):
        if np.isclose(np.linalg.norm(v1 - v2), 2.0 * x):
            x1, y1, z1 = v1
            x2, y2, z2 = v2
            ax.plot3D([x1, x2], [y1, y2], [z1, z2], color="black", linestyle="--", zorder=6, linewidth=1.0)

    fig.savefig("pca.pdf", bbox_inches="tight")


if __name__ == "__main__":

    main()
