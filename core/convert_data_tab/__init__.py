# core/convert_data_tab/__init__.py
# empty or with just a docstring
"""Convert-data tab package."""


# # Maps mode → { role_label: [channels] }
# self.mode_species_layouts = {
#     "tmsd": {
#         # Reaction: Ax + B -> AB + x
#         "Product (AB)": ["Measured"],          # Product in measured channel
#         "Blocked Substrate (Ax)": ["Measured"],# Blocked substrate in measured channel
#     },
#     "hmsd": {
#         # Reaction: A1BB + A2 -> A1A2 + BB
#         "Product (A1A2)": ["Donor", "Acceptor", "FRET"],
#         "Substrate 1 (A1BB)": ["Donor", "Acceptor", "FRET"],
#         "Substrate 2 (A2)": ["Donor", "Acceptor", "FRET"],
#     },
#     "cat3x3": {
#         # Two-step catalytic turnover:
#         # A1x + BB -> A1BB + x
#         # A1BB + A2 -> A1A2 + BB
#         "Product (A1A2)": ["Donor", "Acceptor", "FRET"],
#         "Intermediate (A1BB)": ["Donor", "Acceptor", "FRET"],
#         "Substrate (A2)": ["Donor", "Acceptor", "FRET"],
#         "Blocked Substrate (A1x)": ["Donor", "Acceptor", "FRET"],
#     },
# }
