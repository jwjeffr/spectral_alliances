from pathlib import Path
from itertools import product

import numpy as np
import folium

from clustering import spectral_clustering


def main():

    with Path("alliances.txt").open("r") as file:
        countries = file \
            .read() \
            .replace("\n", ", ") \
            .strip(", ") \
            .split(", ")

    unique_countries = list(set(countries))
    unique_countries.sort()

    index_to_country = {i: country for i, country in enumerate(unique_countries)}
    country_to_index = {y: x for x, y in index_to_country.items()}

    num_countries = len(unique_countries)
    adjacency_matrix = np.zeros((num_countries, num_countries), dtype=float)

    with Path("alliances.txt").open("r") as file:

        for line in file:
            countries = line.strip("\n").split(", ")

            for first_country, second_country in product(countries, repeat=2):
                if first_country == second_country:
                    continue
                i, j = country_to_index[first_country], country_to_index[second_country]
                adjacency_matrix[i, j] = 1.0

    adjacency_matrix /= np.linalg.norm(adjacency_matrix, axis=1)[:, np.newaxis]

    categories = spectral_clustering(adjacency_matrix, num_categories=3)
    country_to_category = {
        country: int(categories[country_to_index[country]]) for country in unique_countries
    }
    colors = {0: "#ff5733", 1: "#a8ffa6", 2: "#faa6ff"}

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


if __name__ == "__main__":

    main()
