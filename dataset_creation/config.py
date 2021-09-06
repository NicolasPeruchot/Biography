"""Configuration for the dataset creation."""

matchers = {
    "PERSON": [
        [
            {"RIGHT_ID": "nom", "RIGHT_ATTRS": {"DEP": "nsubj"}},
            {"LEFT_ID": "nom", "REL_OP": ">>", "RIGHT_ID": "reste", "RIGHT_ATTRS": {}},
        ]
    ],
    "JOB1": [
        [
            {"RIGHT_ID": "nom", "RIGHT_ATTRS": {"DEP": "attr"}},
            {
                "LEFT_ID": "nom",
                "REL_OP": ">",
                "RIGHT_ID": "reste",
                "RIGHT_ATTRS": {"DEP": "compound"},
            },
        ]
    ],
}
