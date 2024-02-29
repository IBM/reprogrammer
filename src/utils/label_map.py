cifar10_labels = [
    lambda c: f'a photo of a {c}',
]

imagenet_labels = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]

cifar2_classnames = ['airplane', 'cat']

cifar10_classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifar100_classnames = ['apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 
    'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 
    'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 
    'hamster', 'house', 'kangaroo', 'computer keyboard', 'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 
    'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak tree', 'orange', 
    'orchid', 'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 
    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 
    'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm']

imagenet100_classnames = [
    "chambered nautilus", "harvestman", "macaw", "bittern bird", "electric ray", "duck",
    "agama", "night snake", "indigo bunting", "tiger shark", "flamingo", "garter snake",
    "smooth newt", "tench", "sulphur-crested cockatoo", "white stork", "terrapin", "eastern diamondback rattlesnake",
    "flatworm", "sea snake", "spiny lobster", "smooth green snake", "bald eagle", "kite (bird of prey)",
    "goldfish", "scorpion", "goose", "bulbul", "chickadee", "sea anemone", "magpie", "mud turtle",
    "crayfish", "lorikeet", "European garden spider", "spoonbill", "hermit crab", "great white shark",
    "green mamba", "bee eater", "bustard", "Komodo dragon", "eastern hog-nosed snake", "green iguana",
    "kingsnake", "toucan", "peafowl", "barn spider", "wolf spider", "worm snake", "American dipper",
    "Dungeness crab", "nematode", "vine snake", "wombat", "southern black widow", "oystercatcher",
    "black grouse", "dunlin", "goldfinch", "snail", "hammerhead shark", "spotted salamander",
    "American alligator", "banded gecko", "wallaby", "great grey owl", "jellyfish", "black swan",
    "ptarmigan", "hummingbird", "desert grassland whiptail lizard", "sidewinder rattlesnake",
    "hen", "Saharan horned viper", "albatross", "axolotl", "tailed frog", "American coot",
    "loggerhead sea turtle", "common redshank", "crane bird", "tick", "sea lion", "tarantula",
    "boa constrictor", "conch", "prairie grouse", "pelican", "coucal", "limpkin", "chiton",
    "yellow garden spider", "leatherback sea turtle", "stingray", "rock crab", "European green lizard",
    "sea slug", "hornbill", "sulphur-crested cockatoo",
]

imagenet_classnames = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
    "American alligator", "triceratops", "worm snake", "ring-necked snake",
    "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
    "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
    "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
    "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
    "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
    "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
    "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
    "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
    "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
    "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
    "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
    "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
    "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
    "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
    "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
]

imagenet_a_indices = [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89,
    90, 94, 96, 97, 99, 105, 107, 108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235,
    254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 
    324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404, 407, 411, 
    416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496,
    514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 
    609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 
    752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 
    831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 
    933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988
]

imagenet_a_classnames = ['stingray', 'goldfinch', 'junco', 'American robin', 'jay', 'bald eagle', 'vulture', 
    'newt', 'American bullfrog', 'box turtle', 'green iguana', 'agama', 'chameleon', 'American alligator', 
    'garter snake', 'harvestman', 'scorpion', 'tarantula', 'centipede', 'sulphur-crested cockatoo', 
    'lorikeet', 'hummingbird', 'toucan', 'duck', 'goose', 'koala', 'jellyfish', 'sea anemone', 'flatworm', 
    'snail', 'crayfish', 'hermit crab', 'flamingo', 'great egret', 'oystercatcher', 'pelican', 'sea lion', 
    'Chihuahua', 'Golden Retriever', 'Rottweiler', 'German Shepherd Dog', 'pug', 'red fox', 'Persian cat', 
    'lynx', 'lion', 'American black bear', 'mongoose', 'ladybug', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 
    'ant', 'grasshopper', 'stick insect', 'cockroach', 'praying mantis', 'leafhopper', 'dragonfly', 
    'monarch butterfly', 'small white butterfly', 'gossamer-winged butterfly', 'starfish', 'cottontail rabbit', 
    'porcupine', 'fox squirrel', 'marmot', 'bison', 'skunk', 'armadillo', 'baboon', 'white-headed capuchin', 
    'African bush elephant', 'pufferfish', 'academic gown', 'accordion', 'acoustic guitar', 'airliner', 
    'ambulance', 'apron', 'balance beam', 'balloon', 'banjo', 'barn', 'wheelbarrow', 'basketball', 'lighthouse', 
    'beaker', 'bikini', 'hunting bow', 'bow tie', 'breastplate', 'broom', 'candle', 'canoe', 'castle', 'cello', 
    'chain', 'storage chest', 'Christmas stocking', 'cowboy boot', 'cradle', 'rotary dial telephone', 
    'digital clock', 'doormat', 'drumstick', 'dumbbell', 'envelope', 'feather boa', 'flagpole', 'forklift', 
    'fountain', 'garbage truck', 'goblet', 'go-kart', 'golf cart', 'grand piano', 'hair dryer', 'clothes iron', 
    'carved pumpkin', 'jeep', 'kimono', 'lighter', 'limousine', 'manhole cover', 'maraca', 'marimba', 'mask', 
    'mitten', 'mosque', 'metal nail', 'obelisk', 'ocarina', 'pipe organ', 'parachute', 'parking meter', 'piggy bank', 
    'pool table', 'hockey puck', 'quill', 'racket', 'fishing casting reel', 'revolver', 'rocking chair', 'rugby ball', 
    'salt shaker', 'sandal', 'saxophone', 'school bus', 'schooner', 'sewing machine', 'shovel', 'sleeping bag', 
    'snowmobile', 'snowplow', 'soap dispenser', 'spatula', 'spider web', 'steam locomotive', 'stethoscope', 'couch', 
    'submarine', 'sundial', 'suspension bridge', 'syringe', 'tank', 'teddy bear', 'toaster', 'torch', 'tricycle', 
    'umbrella', 'unicycle', 'viaduct', 'volleyball', 'washing machine', 'water tower', 'wine bottle', 'shipwreck', 
    'guacamole', 'pretzel', 'cheeseburger', 'hot dog', 'broccoli', 'cucumber', 'bell pepper', 'mushroom', 'lemon', 
    'banana', 'cherimoya (custard apple)', 'pomegranate', 'carbonara', 'bubble', 'cliff', 'volcano', 'baseball player', 
    'rapeseed', "yellow lady's slipper", 'corn', 'acorn'
]

imagenet_r_indices = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94,
    96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 
    163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 
    254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 
    310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 
    365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 
    448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 
    594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 
    805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 
    937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988
]

imagenet_r_classnames = ['goldfish', 'great white shark', 'hammerhead shark', 'stingray', 'hen', 'ostrich', 'goldfinch', 
    'junco', 'bald eagle', 'vulture', 'smooth newt', 'axolotl', 'tree frog', 'green iguana', 'chameleon', 'Indian cobra', 
    'scorpion', 'tarantula', 'centipede', 'peafowl', 'lorikeet', 'hummingbird', 'toucan', 'duck', 'goose', 'black swan', 
    'koala', 'jellyfish', 'snail', 'American lobster', 'hermit crab', 'flamingo', 'great egret', 'pelican', 'king penguin', 
    'grey whale', 'killer whale', 'sea lion', 'Chihuahua', 'Shih Tzu', 'Afghan Hound', 'Basset Hound', 'Beagle', 'Bloodhound', 
    'Italian Greyhound', 'Whippet', 'Weimaraner', 'Yorkshire Terrier', 'Boston Terrier', 'Scottish Terrier', 
    'West Highland White Terrier', 'Golden Retriever', 'Labrador Retriever', 'Cocker Spaniel', 'collie', 'Border Collie', 
    'Rottweiler', 'German Shepherd Dog', 'Boxer', 'French Bulldog', 'St. Bernard', 'Siberian Husky', 'Dalmatian', 'pug', 
    'Pomeranian', 'Chow Chow', 'Pembroke Welsh Corgi', 'Toy Poodle', 'Standard Poodle', 'grey wolf', 'hyena', 'red fox', 
    'tabby cat', 'leopard', 'snow leopard', 'lion', 'tiger', 'cheetah', 'polar bear', 'meerkat', 'ladybug', 'fly', 'bee', 
    'ant', 'grasshopper', 'cockroach', 'praying mantis', 'dragonfly', 'monarch butterfly', 'starfish', 'cottontail rabbit', 
    'porcupine', 'fox squirrel', 'beaver', 'guinea pig', 'zebra', 'pig', 'hippopotamus', 'bison', 'gazelle', 'llama', 'skunk', 
    'badger', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'giant panda', 'eel', 'clownfish', 'pufferfish', 
    'accordion', 'ambulance', 'assault rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub', 'lighthouse', 
    'beer glass', 'binoculars', 'birdhouse', 'bow tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon', 'canoe', 'carousel', 
    'castle', 'mobile phone', 'cowboy hat', 'electric guitar', 'fire truck', 'flute', 'gas mask or respirator', 'grand piano', 
    'guillotine', 'hammer', 'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 'lab coat', 'lawn mower', 'lipstick', 'mailbox', 
    'missile', 'mitten', 'parachute', 'pickup truck', 'pirate ship', 'revolver', 'rugby ball', 'sandal', 'saxophone', 'school bus', 
    'schooner', 'shield', 'soccer ball', 'space shuttle', 'spider web', 'steam locomotive', 'scarf', 'submarine', 'tank', 
    'tennis ball', 'tractor', 'trombone', 'vase', 'violin', 'military aircraft', 'wine bottle', 'ice cream', 'bagel', 'pretzel', 
    'cheeseburger', 'hot dog', 'cabbage', 'broccoli', 'cucumber', 'bell pepper', 'mushroom', 'Granny Smith apple', 'strawberry', 
    'lemon', 'pineapple', 'banana', 'pomegranate', 'pizza', 'burrito', 'espresso', 'volcano', 'baseball player', 'scuba diver', 'acorn'
]