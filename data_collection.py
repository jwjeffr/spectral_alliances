from pathlib import Path
from dataclasses import dataclass
from typing import List, Mapping
from itertools import product

import numpy as np
from numpy.typing import NDArray


ALLIANCES_PATH = Path("alliances.txt")


@dataclass
class Data:

    alliances = Path("alliances.txt")

    @property
    def unique_countries(self) -> List[str]:

        with self.alliances.open("r") as file:
            countries = file \
                .read() \
                .replace("\n", ", ") \
                .strip(", ") \
                .split(", ")

        unique_countries = list(set(countries))
        unique_countries.sort()
        return unique_countries

    @property
    def num_countries(self) -> int:

        return len(self.unique_countries)

    @property
    def index_to_country(self) -> Mapping[int, str]:

        return {i: country for i, country in enumerate(self.unique_countries)}

    @property
    def country_to_index(self) -> Mapping[str, int]:

        return {y: x for x, y in self.index_to_country.items()}

    def get_adjacency_matrix(self) -> NDArray:

        adjacency_matrix = np.zeros((self.num_countries, self.num_countries), dtype=float)

        with Path("alliances.txt").open("r") as file:

            for line in file:
                countries = line.strip("\n").split(", ")

                for first_country, second_country in product(countries, repeat=2):
                    if first_country == second_country:
                        continue
                    i, j = self.country_to_index[first_country], self.country_to_index[second_country]
                    adjacency_matrix[i, j] = 1.0

        adjacency_matrix /= np.sum(adjacency_matrix, axis=1)[:, np.newaxis]
        return adjacency_matrix
