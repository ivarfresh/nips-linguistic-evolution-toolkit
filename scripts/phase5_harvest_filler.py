"""Phase 5 — write 5 Wikipedia-style filler paragraphs as the `s_filler` seed pool.

Hand-curated paragraphs from neutral, game-unrelated topics — pasted text is
adapted from Wikipedia content (paraphrased / trimmed for length). Each is
trimmed to roughly the same length as the s_start myths (~208 words) so
filler vs myth contrasts the seed content, not the seed length.

Writes the pool into data/phase3/seed_manifest.json under `seeds.s_filler`.
"""

import json
import re
import statistics
from pathlib import Path


MANIFEST_PATH = Path("data/phase3/seed_manifest.json")


FILLER_TEXTS = {
    "Brazil": """\
Brazil is the largest country in both South America and Latin America. It covers an area of 8.5 million square kilometres and has over 211 million people. The country is composed of 26 states and the Federal District. Its capital is Brasília, and its most populous city is São Paulo. Brazil is the only country in the Americas to have Portuguese as an official language. It is one of the most multicultural and ethnically diverse nations, due to over a century of mass immigration from around the world. Its Amazon basin includes a vast tropical forest, home to diverse wildlife, a variety of ecological systems, and extensive natural resources spanning numerous protected habitats. This unique environmental heritage positions Brazil as one of the seventeen megadiverse countries on Earth, with the planet's largest reserve of mahogany. To the east, the country has the Atlantic Ocean coastline, which extends for thousands of kilometres along its territory. The Brazilian coast is composed of numerous beaches, mountains, and rivers, attracting visitors from many regions throughout each year. Brazil's economy includes a mix of agriculture, mining, manufacturing, and services. It is a member of various international organizations including BRICS.""",
    "Octopus": """\
An octopus is a soft-bodied, eight-limbed mollusc of the order Octopoda. Around 300 species are recognised, and the order is grouped within the class Cephalopoda with squids, cuttlefish, and nautiloids. Like other cephalopods, an octopus is bilaterally symmetric with two eyes and a beaked mouth at the centre point of the eight limbs. The soft body can radically alter its shape, enabling octopuses to squeeze through small gaps. They trail their eight appendages behind them as they swim. The siphon is used for respiration and locomotion, by expelling a jet of water. Octopuses have a complex nervous system and excellent sight, and are among the most intelligent and behaviourally diverse of all invertebrates. They inhabit various regions of the ocean, including coral reefs, pelagic waters, and the seabed; some live in the intertidal zone, and others live at great depths. Most species grow quickly, mature early, and are short-lived. In most species, the male uses a specially adapted arm to deliver a bundle of sperm directly into the female's mantle cavity, after which he becomes senescent and dies. The female deposits fertilised eggs in a den and cares for them until they hatch, after which she also dies.""",
    "Copper": """\
Copper is a chemical element with the symbol Cu and atomic number 29. It is a soft, malleable, and ductile metal with very high thermal and electrical conductivity. A freshly exposed surface of pure copper has a pinkish-orange color. Copper is used as a conductor of heat and electricity, as a building material, and as a constituent of various metal alloys, such as sterling silver used in jewelry, cupronickel used to make marine hardware and coins, and constantan used in strain gauges and thermocouples for temperature measurement. Copper is one of the few metals that can occur in nature in a directly usable metallic form. This led to very early human use in several regions, from circa 8000 BC. Thousands of years later, it was the first metal to be smelted from sulfide ores, circa 5000 BC; the first metal to be cast into a shape in a mold, circa 4000 BC; and the first metal to be purposely alloyed with another metal, tin, to create bronze, circa 3500 BC. In the Roman era, copper was mined principally on Cyprus, the origin of the name of the metal, from aes Cyprium.""",
    "Glass": """\
Glass is a non-crystalline, often transparent, amorphous solid that has widespread practical, technological, and decorative use in tableware, optics, and windows. Glass is most often formed by rapid cooling of the molten form; some glasses such as volcanic glass are naturally occurring. The most familiar, and historically the oldest, types of manufactured glass are silicate glasses based on the chemical compound silica, the primary constituent of sand. Soda-lime glass, containing around 70 percent silica, accounts for around 90 percent of manufactured glass. The term glass, in popular usage, is often used to refer only to this type of material, although silica-free glasses often have desirable properties for applications in modern communications technology. Some objects, such as drinking glasses and eyeglasses, are so commonly made of silicate-based glass that they are simply called by the name of the material. Although brittle, buried silicate glass will survive for very long periods if not disturbed, and many examples of glass fragments exist from early glass-making cultures. Archaeological evidence suggests glass-making dates back to at least 3600 BC in Mesopotamia, Egypt, or Syria. The earliest known glass objects were beads, perhaps created accidentally during metalworking or the production of faience.""",
    "Photosynthesis": """\
Photosynthesis is a system of biological processes by which photosynthetic organisms, such as most plants, algae, and cyanobacteria, convert light energy, typically from sunlight, into the chemical energy necessary to fuel their metabolism. Photosynthesis usually refers to oxygenic photosynthesis, a process that produces oxygen. Photosynthetic organisms store the chemical energy so produced within intracellular organic compounds like sugars, glycogen, cellulose, and starches. To use this stored chemical energy, an organism's cells metabolize the organic compounds through cellular respiration. Photosynthesis plays a critical role in producing and maintaining the oxygen content of the Earth's atmosphere, and it supplies most of the biological energy necessary for complex life on Earth. Some bacteria also perform anoxygenic photosynthesis, which uses bacteriochlorophyll to split hydrogen sulfide as a reductant instead of water, producing sulfur instead of oxygen. Archaea such as Halobacterium also perform a type of non-carbon-fixing anoxygenic photosynthesis, where the simpler photopigment retinal and its microbial rhodopsin derivatives are used to absorb green light. Most plants, most algae, and cyanobacteria perform oxygenic photosynthesis, which uses water as a reductant. In this process, water is oxidized, producing molecular oxygen as a byproduct.""",
}


def word_count(text):
    return len(re.findall(r"\b[a-zA-Z']+\b", text))


def harvest():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    filler_seeds = []
    for topic, text in FILLER_TEXTS.items():
        text = " ".join(text.split())  # collapse whitespace
        words = word_count(text)
        filler_seeds.append({
            "source_run": "wikipedia_paraphrased",
            "agent_id": None,
            "round": None,
            "joint_at_source": None,
            "text": text,
            "tokens": words,
            "topic": topic,
        })
        print(f"  {topic}: {words} words")

    manifest["seeds"]["s_filler"] = filler_seeds

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(filler_seeds)} filler seeds to {MANIFEST_PATH}")
    print(f"Mean word count: {statistics.mean(s['tokens'] for s in filler_seeds):.1f}")
    print(f"vs. s_start mean (208.0)")


if __name__ == "__main__":
    harvest()
