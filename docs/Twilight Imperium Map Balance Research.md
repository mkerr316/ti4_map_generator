

# **Balancing the Galaxy: An Analysis of Spatial Statistics and Fairness in Twilight Imperium 4 Map Generation**

## **Executive Summary**

This report conducts a technical analysis of current map generation tools for the board game *Twilight Imperium: Fourth Edition* (TI4), evaluates their balancing methodologies against principles of game theory and spatial analysis, and proposes a new, superior framework for procedural map generation.  
Current generators, while effective at streamlining game setup 1, operate on balancing principles that are quantifiably insufficient. The analysis reveals an evolution from simple, user-defined aggregations to more complex heuristics, such as the "Optimal Value" metric used in the popular "Milty Draft" system.2 The most advanced generators even employ rudimentary spatial metrics, such as an "inverse distance squared" calculation, to approximate resource access.3  
However, these methods fundamentally fail to model the critical spatial relationships that define the *TI4* gameplay experience. They are, in effect, balancing a spreadsheet of tile values rather than the holistic, spatial game. This omission creates a significant, measurable "balance loss" by ignoring two key components of spatial statistics: **Point Pattern Analysis** (e.g., the clustering of resources and anomalies) and **Accessibility Models** (e.g., the pathfinding cost to strategic objectives).  
This report concludes that the next generation of map generators *must* incorporate these formal spatial statistics. By redefining "slice balance" as a multi-metric function that includes quantitative value, resource clustering, and objective accessibility, it is possible to move from balancing abstract *parts* to balancing the *whole, perceived game*. This is not merely an improvement but a fundamental necessity to solve the "perceived unfairness" 4 that current generators inadvertently create.

## **I. The Current Paradigm: A Technical Review of TI4 Map Generation Algorithms**

The community of *Twilight Imperium 4* players has developed several map generation tools to replace the time-consuming and often unbalanced rulebook method of map creation.5 An analysis of these generators reveals a clear, albeit incomplete, evolution from simple summation to rudimentary spatial awareness.

### **A. Simple Aggregation (The "Spreadsheet" Approach)**

The baseline for modern generators, such as the popular KeeganW tool 1, primarily functions as a convenience and randomization tool. While its goal is to "cut down game setup time" and generate a "fair and balanced map" 1, its mechanism for achieving this "balance" is a non-spatial tally configured by the user.  
The generator's balance logic is exposed through user-configurable "weights" for "Resource Influence," "Planet Count," "Specialty," "Anomaly," and "Wormhole".1 This design places the *onus of balance* entirely on the user. The tool does not *solve* the complex balance problem; it merely provides a high-speed calculator for executing a user's (potentially flawed) *theory* of balance. The generator itself is "balance-agnostic," incapable of distinguishing a strategically sound map from one that is merely numerically plausible.

### **B. Heuristic-Based Metrics (The "Milty Draft" Logic)**

The "Milty Draft" system 2 is the *de facto* standard for many competitive player groups and represents a more nuanced approach. It employs a two-pronged balancing methodology.  
First, it uses a superior quantitative heuristic known as **"Optimal Value."** Instead of simply summing a planet's resources and influence, this metric calculates a planet's value as the *higher* of its resource or influence, treating the other value as zero (e.g., a 3/1 planet is valued as 3/0).2 This "Optimal Value" heuristic is a direct acknowledgment of the "rich but poor" problem often discussed by players.9 It correctly identifies that a 2/2 planet is not "worth" 4 (as a simple tally would suggest) but is instead "worth" its most *efficient* use, which is typically its resource value.10 While a superior model to simple aggregation, it remains a non-spatial heuristic.  
Second, the Milty Draft algorithm employs a *categorical* heuristic: the **"3-Tier System."** The algorithm designers, recognizing that "Optimal Value" alone is insufficient, categorize all blue-backed tiles into three tiers of quality.2 Each generated 5-tile "slice" is then dealt one tile from each tier, plus two red-backed (anomaly/empty) tiles.2 The specific criteria for these tiers are not publicly defined 12, functioning as a "black box" form of curated content. This 3-tier system is a *categorical fix* for the failures of the *quantitative* heuristic. It prevents the creation of a slice with a high "Optimal Value" score that is composed entirely of "Tier 3" (presumably low-value) tiles, which would *feel* terrible to play. It forces a more equitable distribution of "good" systems.

### **C. Rudimentary Spatial Metrics (The "Inverse Distance" Model)**

The "TI4 Balanced Map Generator" by Derek Peterson represents the most advanced *current* model.3 Its stated goal is to "provide players with approximately equal access to resources, influence and tech specialties".3  
The core of this generator is a "balance score" calculated by allocating a portion of each tile's value to a given home system. This allocation is "proportional to the **inverse distance squared** of the home system to the tile in question".3 The algorithm then calculates the *variances* of these total allocated sums between all home systems and iteratively swaps tiles to minimize this variance.3  
This "inverse distance squared" model is a direct, if simplified, implementation of the **Gravity Model** from spatial statistics.13 This connection is crucial: it confirms that generator designers are *already* attempting to use spatial statistics to solve the balance problem. The issue is that this model is rudimentary. Its "distance" metric is basic (though it attempts to account for wormholes 3), and it fails to model *network* distance (e.g., the true pathfinding cost through anomalies) or *clustering* (a "hot spot" of resources three tiles away is strategically different from three single resources dispersed three tiles away).

### **Table 1: Comparison of TI4 Map Generator Balancing Algorithms**

| Generator | Primary Quantitative Metric | Spatial Balancing Component | Core Limitation |
| :---- | :---- | :---- | :---- |
| **KeeganW Generator** 1 | User-Weighted Sums | None (User-defined weights) | Balance is user-dependent; generator is a calculator. |
| **Milty Draft** 2 | "Optimal Value" Tally | Categorical (3-Tier System) | "Optimal Value" is non-spatial; slice-based assembly creates map-level spatial faults. |
| **DerekPeterson Generator** 3 | Inverse-Distance-Squared Allocated Sum | Quantitative (Inverse Distance Squared) | "Distance" metric is oversimplified; fails to account for resource clustering or true network path cost. |

## **II. The Player's Verdict: Community Definitions of "Slice" Quality and Balance Failures**

A significant disconnect exists between the *technical* implementation of "balance" in these generators and the *perceived* outcomes experienced by players. Community discussions provide a rich qualitative dataset that establishes a *holistic, player-centric* definition of balance that current generators fail to capture.

### **A. Defining the "Holistic Slice": Beyond the Tile Tally**

While the technical definition of a "slice" is simply the 4-5 tiles closest to a player's home system 10, the *player's* definition is not a static list of tiles but a *dynamic assessment of strategic potential*. This potential is defined by non-quantitative, spatial, and strategic factors that current generators largely ignore.  
Key factors identified by players include:

1. **"Path to Mecatol Rex":** The ease of access to the central planet is a primary driver of mid-game strategy.15 A path blocked by a supernova or nebula is a significant strategic disadvantage.17  
2. **"Forward Dock Location":** A "high resource planet forward of home" 18 is a critical, purely spatial asset for unit projection and board control.  
3. **Wormhole Adjacency:** A wormhole can be a "back door" for an opponent or a strategic highway for the player, drastically altering the map's effective geography.18  
4. **Tech Skip Distribution:** The *type* (e.g., blue, green) and *location* (e.g., safe vs. equidistant) of tech skips are often valued more than raw resources.10

### **B. Generator Failure Cases: The "Feel-Bad" Slice**

Players consistently report that generators produce "unbalanced" or "feel-bad" maps.14 The specific complaints are overwhelmingly *spatial*, not quantitative. They include "anomaly next to anomaly," "wormholes leading nowhere," and "resource distribution being all over the shop".24  
This reveals a fundamental disconnect. The Milty Draft generator, for example, explicitly states that "slices cannot contain... anomalies next to one another".2 Yet, players complain of exactly this problem.24 The generator's own documentation provides the reason: "Anomalies may be placed next to one another if they were on the borders of two slices, due to the way slices are assembled".2  
This is a *critical failure* of the slice-based generation model. The generator successfully balances the *slice* as an abstract object. However, when these "balanced" *parts* are assembled, they form an "unbalanced" *whole*. The system ensures local balance at the cost of global, map-wide spatial integrity.

### **C. The "Rich but Poor" Problem: Quantifying Strategic Poverty**

The most sophisticated community critique is the "rich but poor" slice.9 A slice can be numerically "rich" according to the generator's metrics but strategically "poor."

* **Example 1: Resource-Type Imbalance.** A slice with a high "Optimal Value" score that is comprised *only* of influence-heavy planets is "influence-strapped" 25 and described as "dogshit" 22 for most factions, as it lacks the resources to build fleets.  
* **Example 2: Spatial-Distribution Imbalance.** A slice valued at 8 "Optimal Value" is perceived very differently if that value is dispersed over four safe, easily-controlled planets versus being clustered in one high-value planet located in a contested equidistant system.18

This "rich but poor" problem reveals the core "balance" we are losing. The *actual* value of a slice is a function: $Value \= (Sum \\, of \\, Optimal \\, Values) \\times (Probability \\, of \\, Control)$. The *Probability of Control* is a spatial function of distance, adjacency, anomalies, and chokepoints. By ignoring this probability, current generators *systematically over-value* high-risk, high-clustering slices and *under-value* safer, dispersed slices. They are calculating the *theoretical maximum* value, not the *practical, spatially-constrained* value.

## **III. A Game-Theoretic Framework for Fairness in Asymmetric 4X Games**

To properly diagnose this "balance loss," we must first establish a rigorous, theoretical definition of "balance" and "fairness" in the context of a complex, asymmetric, N-player, non-zero-sum game like *Twilight Imperium*.26

### **A. Distinguishing "Balance," "Fairness," and "Equity"**

In game design theory, these are distinct terms.28

* **Balance:** The adjustment of game elements (e.g., factions, technologies) to meet a designer's intended level of power and viability.31 Balance is *not* equality; it is about creating *interesting decisions*.31  
* **Fairness:** The principle that all players have a "roughly... same chance of winning at the start independent of which offered options they choose".29 A game can be *unbalanced* (e.g., *Halo*'s combat encounters) but must be *fair*.29  
* **Equity:** The *perceived* fairness of a resource distribution, which is often culturally or contextually defined.30

The map generation problem is not about *balance* (which would involve faction powers) but about *fairness*. The goal is to provide each player with an *equal opportunity* at the start, *before* their asymmetric factions and relative skill modify the outcome.

### **B. "Process Fairness" vs. "Outcome Fairness"**

Academic literature on game design provides a crucial distinction between two types of fairness 33:

* **Outcome Fairness:** Defined by the final win ratio (e.g., $p \= 0.5$).33 This is an impossible and undesirable goal for a complex, 6-player, skill-based game like TI4.  
* **Process Fairness:** Defined by the initial distribution of advantages and disadvantages. A game is considered fair if the *number* of advantages and disadvantages are roughly equal for all participants at the outset.33

*Twilight Imperium* map generation is a problem of *Process Fairness*. The goal is not to guarantee every player a 1/6 chance of winning, but to ensure that no player *starts* with a significant, unearned *disadvantage* (like a "dogshit slice" 22) or *advantage* (a "stupid rich" slice 9).

### **C. Perceived Fairness vs. Mathematical Fairness**

The *Wolfenstein: Enemy Territory* anecdote is a canonical text in game design.34 In the game, players complained that one faction's Thompson submachine gun was more powerful than the other's MP40. Statistical analysis of player data proved them *correct*: the Thompson had a higher kill rate. However, a technical analysis of the game's code showed the stats for the weapons were *identical*.  
The *cause* of this real, statistical imbalance was *perceived* factors: the Thompson's 3D model and its "bass-y" sound *felt* more powerful, which led players to use it more confidently and effectively.34  
This is *directly applicable* to TI4 map generation. A slice that *looks* bad (e.g., an ugly anomaly blocking the path to Mecatol Rex 17) or *feels* poor (e.g., all influence-planets 25) *will be perceived as "unbalanced"*.10 This perception may lead to a self-fulfilling prophecy of poor play, *even if its 'Optimal Value' is mathematically identical to a 'good' slice*. The *spatial arrangement* of the map *is* the "sound and model" of the game. A purely quantitative generator will *always* fail because it ignores this critical *perceptual* and *psychological* component of fairness.34

### **D. Formalizing Fairness: α-fairness and Jain's Index**

Resource allocation theory provides formal, axiomatic models for measuring fairness.36

* **$\\alpha$-fairness:** This framework defines a trade-off between *efficiency* (maximizing the *sum* of all resources) and *fairness* (maximizing the *minimum* allocated resource, i.e., "max-min"). A "larger $\\alpha$" means the system values fairness more than efficiency.36  
* **Jain's Index:** This is a specific fairness measure that quantifies the equality of a resource distribution, producing a score from 0 (total inequality) to 1 (total equality).36

This provides the central theoretical tool to solve the map-balancing problem. Current generators (as analyzed in Section I) are balancing for *efficiency* (the *sum* of each slice). A *true* balancing algorithm must balance for *fairness*.  
The correct approach is to model the entire map as a resource allocation vector $X \= \[v\_1, v\_2,..., v\_6\]$, where $v\_i$ is the *holistic value* of player *i*'s slice. The generator's objective function should not be to make each $v\_i$ "high," but to **maximize $Jain's\\\_Index(X)$**. This is a profoundly different and superior methodology. It stops balancing slices in isolation and starts balancing the *entire map* as a *system of relationships*.

## **IV. The Spatial Dimension: A Methodological Primer for Map Analysis**

This section formally introduces the concepts from spatial analysis and statistics that are the missing component in current generators. These tools are required to quantify the spatial relationships that players (in Section II) intuitively understand and complain about.

### **A. Quantifying "Clumping" (Point Pattern Analysis)**

Point Pattern Analysis 37 is a branch of spatial statistics used to determine if the spatial distribution of a phenomenon is clustered, dispersed, or random. In this context, it answers the question: "Are high-value planets clustered together?"

1. *Getis-Ord Gi (Hot Spot Analysis):*\* This statistic identifies statistically significant "hot spots" (clusters of *high* values) and "cold spots" (clusters of *low* values).40 Critically, "a feature with a high value... may not be a statistically significant hot spot. To be... a hot spot, a feature will have a high value *and be surrounded by other features with high values*".43  
   * **Application to TI4:** This tool *quantifies* the "rich but poor" spatial problem. A slice with one major resource "hot spot" (e.g., all its R/I clustered in one or two adjacent tiles) is *less safe* and *less valuable* than a slice with the *same total R/I* but a "cold" Gi\* score (dispersed).  
2. **Moran's I (Spatial Autocorrelation):** This statistic measures "spatial autocorrelation," or whether *similar* values (high/high *or* low/low) are clustered.40 A positive Moran's I indicates clumping; a negative one indicates a "checkerboard" pattern of dissimilar values.42  
   * **Application to TI4:** This can be used to balance *categorical* features. Are all the tech skips (a categorical value) "clumped" in one quadrant of the map (high positive Moran's I)? Are all the anomalies 24 "clumped" together? This statistic *directly* measures the community's spatial complaints.

### **B. Quantifying "Accessibility" (Network & Gravity Models)**

This branch of spatial analysis measures the *friction* of moving between points.

1. **The Gravity Model:** This model is the workhorse of accessibility analysis.13 It states that the interaction (or accessibility) between two points is a function of their "Mass" divided by their "Distance," often with an exponent: $Accessibility \\propto \\frac{Mass}{Distance^{\\beta}}$.13  
   * "Mass" is the *scale* or *attractiveness* of the resource (e.g., a city's population 13 or, for TI4, a planet's *R/I value*).  
   * "Distance" is the *friction* to reach it (e.g., travel time 49 or, for TI4, the *movement cost* in turns/fleet).  
   * **Application to TI4:** This *quantifies* the "path to Mecatol Rex".18 We can calculate an "Accessibility Score to Mecatol" for each slice. Mecatol is the "Mass," and "Distance" is the pathfinding cost. A "blocking" anomaly 17 would make the $Distance$ value very high, thus *quantifiably* reducing that slice's accessibility score.  
2. *Hexagonal Pathfinding (A):*\* The "Distance" in our TI4 Gravity Model cannot be simple Euclidean distance. It must be a *network graph* distance. Pathfinding on a hexagonal grid 50 is a solved problem using standard algorithms like A\* or Dijkstra.52 In this graph, anomalies (asteroids, supernovas) would be nodes with a *high travel cost*, while wormholes would be edges with a *low travel cost*.

These tools from spatial analysis are the missing link. They *quantify* the *intuitive, qualitative* complaints from the player community.

### **Table 2: Spatial Statistics for Game Map Analysis**

| Statistical Tool | What It Measures | Application to TI4 Map |
| :---- | :---- | :---- |
| **Getis-Ord Gi**\* (Hot Spot Analysis) | Identifies statistically significant *local* clusters of *high* or *low* numerical values.40 | Quantifies the "risk" of a slice. A "hot spot" of resources is a high-value, high-risk cluster (a single point of failure), making a slice "rich but poor." |
| **Moran's I** (Spatial Autocorrelation) | Measures *global* "clumping" of *similar* values (e.g., high-high or low-low).40 | Quantifies *categorical* balance. Used to detect if all tech skips, or all anomalies, are "clumped" in one area of the map, (solving the "anomaly next to anomaly" problem 24). |
| **Gravity Model** | Measures "Accessibility" to a point of "Mass" (value) as a function of "Distance" (friction/cost).13 | Quantifies *strategic potential*. Calculates a numerical score for "Path to Mecatol Rex," "Path to Legendary Planets," or "Proximity to Enemies," by using planet values as "Mass" and pathfinding cost as "Distance." |

## **V. A New Model for Map Generation: Holistic Balance Through Spatial Statistics**

This analysis synthesizes into a proposal for a next-generation "holistic" balancing algorithm that directly incorporates spatial statistics.

### **A. The "Holistic Slice Score": A Multi-Metric Evaluation Function**

The core flaw of current generators is that they *assemble* pre-balanced slices, which creates an unbalanced whole.2 A new model must *generate a whole map* first, then *evaluate* it using an iterative process (similar to the method in 3 or the one used for *Terra Mystica* 54).  
The "value" of a slice must be redefined. The "Holistic Slice Score" ($v\_i$) for each player is not a simple sum. It must be a multi-metric function: $v\_i \= f(Q, C, A)$.

1. **$Q$ (Quantitative Value):** The "Optimal Value" 2 of the 4-5 "owned" tiles. This serves as the baseline quantitative score.  
2. **$C$ (Clustering/Safety Score):** The "internal" spatial distribution. A **Getis-Ord Gi**\* 40 analysis is run on the slice's resource values. A high Gi\* (a "hot spot") is *bad* (high risk) and applies a *penalty* to the score. A **Moran's I** 40 analysis is run for anomalies, penalizing "clumped" anomalies.24 This is directly analogous to the map requirements for the *Terra Mystica* generator, which penalizes unfavorable terrain adjacencies.54  
3. **$A$ (Accessibility Score):** The "external" spatial value. A **Gravity Model** 13 is run from the player's home system to key objectives (Mecatol Rex 15, other home systems, Legendary planets). This *quantifies* the "path to Rex" 18 as a positive component of the slice score.

### **B. Adapting the "Six Measures" Framework**

This $f(Q, C, A)$ model is strongly supported by academic procedural content generation (PCG) literature. A 2013 paper on map generation for strategy games 55 defines six "measures of quality" that map directly to our proposed model:

* $Q$ (Quantitative) corresponds to $b\_{res}$ (Resource Balance).  
* $C$ (Clustering/Safety) corresponds to $f\_{res}$ (Resource Safety) and $f\_{saf}$ (Base Safety).  
* $A$ (Accessibility) corresponds to $f\_{exp}$ (Exploration \- pathfinding difficulty).

This alignment demonstrates that the proposed model is consistent with state-of-the-art academic research on balanced map generation.

### **C. The Proposed "Holistic Generator" Algorithm**

A next-generation generator would operate on the following algorithm:

1. **Generate:** Create a (pseudo-random) full map.  
2. **Evaluate (Holistic Slice Score):** For each of the 6 player positions, calculate its Holistic Slice Score $v\_i \= f(Q, C, A)$.  
3. **Evaluate (Map Fairness):** Create the map's resource vector $X \= \[v\_1, v\_2, v\_3, v\_4, v\_5, v\_6\]$.  
4. **Calculate Fairness:** Compute the map's total fairness score: $F\_{map} \= Jain's\\\_Index(X)$.36  
5. **Iterate:** If $F\_{map}$ is below a set threshold (e.g., \< 0.95), perform a "mutation" (e.g., swap two tiles, as in 3) and return to Step 2\. Repeat until a "fair" map is found.

This model *quantifies* and *solves* the community's qualitative complaints. The "rich but poor" slice 22 gets a low $v\_i$ score (via the $Q$ or $C$ component). The "blocked path" slice 17 gets a low $v\_i$ (via the $A$ component). The "anomaly next to anomaly" map 24 gets a low $v\_i$ (via the $C$ component). This model transforms the "feel-bad" elements 10 into quantifiable penalties.

## **VI. The Cost of Omission: Measuring the Balance We Are Losing**

This analysis directly answers the final question: "How much balance are we losing by not considering spatial distribution?" The answer is that we are losing the very *nature* of the game we are trying to balance.

### **A. The Nature of the Loss: Strategic Depth**

The "balance" we are losing is *strategic depth*.31 Game design theory posits that "balance is depth".31 When a generator creates a "dominant strategy" (e.g., one slice is demonstrably *better* because its spatial layout is superior, even if its *numbers* are average), it *reduces* the depth of the game.  
The "feel bad" slice 22 is one where the player's *strategic agency* is removed by the map's spatial constraints *before the game even begins*. This is the antithesis of a good strategy game, which should be defined by *meaningful decisions*.32

### **B. Quantifying the Loss: The "Spatial Imbalance Delta"**

This loss can be quantified by defining the *delta* (the "Balance Loss") between what current generators *think* is balanced and what *is* balanced.  
$Generator\\\_Imbalance \= Variance(\[sum(s\_1), sum(s\_2),..., sum(s\_6)\])$  
(This is the simple, non-spatial metric current generators try to minimize 3).  
$True\\\_Imbalance \= 1 \- Jain's\\\_Index(\[v\_1, v\_2,..., v\_6\])$  
(This is the true, holistic imbalance, using our holistic $v\_i$ score 36).  
$Balance\\\_Loss \= True\\\_Imbalance \- Generator\\\_Imbalance$  
This "Balance Loss" represents the *perceptual* and *strategic* unfairness that current generators *fail to see*.

### **C. Final Conclusion: Balancing the Game, Not the Spreadsheet**

By not considering spatial distribution, current generators are *not* balancing *Twilight Imperium 4*—a game of movement, logistics, and spatial control.56 They are only balancing a *spreadsheet* (a list of R/I values).  
This omission creates the *exact* perceptual and strategic imbalances 10 they were designed to solve. It is the digital equivalent of the *Wolfenstein* anecdote 34: a game is shipped with "identical stats," but one player's slice "feels" powerful while another's "feels" weak. This *perceived unfairness* 4 is a *real* imbalance.  
The "balance" we are losing is the *entire spatial half* of the game. We are calculating "Optimal Value" 2 but ignoring the "path to Rex" 18, ignoring the *risk* of "hot spots" 40, and ignoring the *friction* of anomalies.17 The introduction of formal spatial statistics is not merely an "improvement"; it is a *fundamental necessity* to correct this omission and create maps that are truly "fair" in the way players perceive and play the game.

#### **Works cited**

1. TI4 Map Generator \- Keegan Williams, accessed November 7, 2025, [https://keeganw.github.io/ti4/](https://keeganw.github.io/ti4/)  
2. Milty Draft: TI4, accessed November 7, 2025, [https://milty.shenanigans.be/](https://milty.shenanigans.be/)  
3. TI4 Balanced Map Generator : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/8w5y8i/ti4\_balanced\_map\_generator/](https://www.reddit.com/r/twilightimperium/comments/8w5y8i/ti4_balanced_map_generator/)  
4. Evolution of fairness in the one-shot anonymous Ultimatum Game \- PNAS, accessed November 7, 2025, [https://www.pnas.org/doi/10.1073/pnas.1214167110](https://www.pnas.org/doi/10.1073/pnas.1214167110)  
5. Tutorial: Guide to Map Making (SCPT) : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/18mfllo/tutorial\_guide\_to\_map\_making\_scpt/](https://www.reddit.com/r/twilightimperium/comments/18mfllo/tutorial_guide_to_map_making_scpt/)  
6. KeeganW/ti4: TI4 Map Generator \- Generate a custom ... \- GitHub, accessed November 7, 2025, [https://github.com/KeeganW/ti4](https://github.com/KeeganW/ti4)  
7. r/twilightimperium Wiki: TI4 Map Resources & Generators \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/wiki/maps/](https://www.reddit.com/r/twilightimperium/wiki/maps/)  
8. Twilight Struggle | TI4 \- Milty Draft, accessed November 7, 2025, [https://milty.shenanigans.be/d/661cdcf31df9e](https://milty.shenanigans.be/d/661cdcf31df9e)  
9. Most balanced average of resources and influence per slice? : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/11qf0ko/most\_balanced\_average\_of\_resources\_and\_influence/](https://www.reddit.com/r/twilightimperium/comments/11qf0ko/most_balanced_average_of_resources_and_influence/)  
10. Can someone explain to me the whole "slice" thing? : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/hh36g1/can\_someone\_explain\_to\_me\_the\_whole\_slice\_thing/](https://www.reddit.com/r/twilightimperium/comments/hh36g1/can_someone_explain_to_me_the_whole_slice_thing/)  
11. Self-Generating IRL Milty Draft Slices : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/ozdc7f/selfgenerating\_irl\_milty\_draft\_slices/](https://www.reddit.com/r/twilightimperium/comments/ozdc7f/selfgenerating_irl_milty_draft_slices/)  
12. Milty Draft Blue Tile Tiers? : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/10yastr/milty\_draft\_blue\_tile\_tiers/](https://www.reddit.com/r/twilightimperium/comments/10yastr/milty_draft_blue_tile_tiers/)  
13. GRAVITY AND SPATIAL INTERACTION MODELS \- Portland State ..., accessed November 7, 2025, [https://web.pdx.edu/\~stipakb/download/PA557/ReadingsPA557sec1-2.pdf](https://web.pdx.edu/~stipakb/download/PA557/ReadingsPA557sec1-2.pdf)  
14. How to make a balanced map : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/12e4oqo/how\_to\_make\_a\_balanced\_map/](https://www.reddit.com/r/twilightimperium/comments/12e4oqo/how_to_make_a_balanced_map/)  
15. Is it always worth trying to get Mecatol first if you can? : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/1gi6yuo/is\_it\_always\_worth\_trying\_to\_get\_mecatol\_first\_if/](https://www.reddit.com/r/twilightimperium/comments/1gi6yuo/is_it_always_worth_trying_to_get_mecatol_first_if/)  
16. Strategies for completing the mecatol rex secret objectives? \- FFG Forum Archive, accessed November 7, 2025, [https://ffg-forum-archive.entropicdreams.com/topic/29568-strategies-for-completing-the-mecatol-rex-secret-objectives/](https://ffg-forum-archive.entropicdreams.com/topic/29568-strategies-for-completing-the-mecatol-rex-secret-objectives/)  
17. Advice Needed for TI 4 Milty Draft Strategy \- 5 Players, I'm 1st to Pick\! : r/twilightimperium, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/1fsbuep/advice\_needed\_for\_ti\_4\_milty\_draft\_strategy\_5/](https://www.reddit.com/r/twilightimperium/comments/1fsbuep/advice_needed_for_ti_4_milty_draft_strategy_5/)  
18. What to look for when choosing a slice? : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/1bo6he9/what\_to\_look\_for\_when\_choosing\_a\_slice/](https://www.reddit.com/r/twilightimperium/comments/1bo6he9/what_to_look_for_when_choosing_a_slice/)  
19. "4 player balanced map" : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/mj5guc/4\_player\_balanced\_map/](https://www.reddit.com/r/twilightimperium/comments/mj5guc/4_player_balanced_map/)  
20. I tried the "Balanced Map Generator" \- Some issues : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/8ztza2/i\_tried\_the\_balanced\_map\_generator\_some\_issues/](https://www.reddit.com/r/twilightimperium/comments/8ztza2/i_tried_the_balanced_map_generator_some_issues/)  
21. How many slices and factions do you usually use for a Milty draft? \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/r6o6vh/how\_many\_slices\_and\_factions\_do\_you\_usually\_use/](https://www.reddit.com/r/twilightimperium/comments/r6o6vh/how_many_slices_and_factions_do_you_usually_use/)  
22. How to play when your slice is terrible? : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/1nmkkbe/how\_to\_play\_when\_your\_slice\_is\_terrible/](https://www.reddit.com/r/twilightimperium/comments/1nmkkbe/how_to_play_when_your_slice_is_terrible/)  
23. Has Milty Draft Changed Equidistant Meta? In a Bad Way? : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/r6nkcs/has\_milty\_draft\_changed\_equidistant\_meta\_in\_a\_bad/](https://www.reddit.com/r/twilightimperium/comments/r6nkcs/has_milty_draft_changed_equidistant_meta_in_a_bad/)  
24. Is it time to move on from Milty Draft? : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/1mljl0h/is\_it\_time\_to\_move\_on\_from\_milty\_draft/](https://www.reddit.com/r/twilightimperium/comments/1mljl0h/is_it_time_to_move_on_from_milty_draft/)  
25. Best Factions for Influence-Strapped Slices : r/twilightimperium \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/twilightimperium/comments/q2zokd/best\_factions\_for\_influencestrapped\_slices/](https://www.reddit.com/r/twilightimperium/comments/q2zokd/best_factions_for_influencestrapped_slices/)  
26. Balancing Game Levels for Asymmetric Player Archetypes With Reinforcement Learning, accessed November 7, 2025, [https://arxiv.org/html/2503.24099v1](https://arxiv.org/html/2503.24099v1)  
27. Game theory \- Wikipedia, accessed November 7, 2025, [https://en.wikipedia.org/wiki/Game\_theory](https://en.wikipedia.org/wiki/Game_theory)  
28. Game Balancing Strategies & Tips (Design Guide) \- Helika, accessed November 7, 2025, [https://www.helika.io/game-balancing-strategies-tips-design-guide/](https://www.helika.io/game-balancing-strategies-tips-design-guide/)  
29. Game balance \- Wikipedia, accessed November 7, 2025, [https://en.wikipedia.org/wiki/Game\_balance](https://en.wikipedia.org/wiki/Game_balance)  
30. HC431H \- Cooperative Game Theory: An investigation of fairness principles, accessed November 7, 2025, [https://honors.uoregon.edu/hc431h-cooperative-game-theory-investigation-fairness-principles](https://honors.uoregon.edu/hc431h-cooperative-game-theory-investigation-fairness-principles)  
31. Balance is Depth: Why balance matters, even in single player games : r/truegaming \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/truegaming/comments/tuh233/balance\_is\_depth\_why\_balance\_matters\_even\_in/](https://www.reddit.com/r/truegaming/comments/tuh233/balance_is_depth_why_balance_matters_even_in/)  
32. Video Game Balance: A Definitive Guide \- Game Design Skills, accessed November 7, 2025, [https://gamedesignskills.com/game-design/game-balance/](https://gamedesignskills.com/game-design/game-balance/)  
33. What Constitutes Fairness in Games? A Case Study with Scrabble, accessed November 7, 2025, [https://www.mdpi.com/2078-2489/12/9/352](https://www.mdpi.com/2078-2489/12/9/352)  
34. Map balance | The Level Design Book, accessed November 7, 2025, [https://book.leveldesignbook.com/process/combat/balance](https://book.leveldesignbook.com/process/combat/balance)  
35. What's the mathematics behind balancing a game system? : r/rpg \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/rpg/comments/yf1gbl/whats\_the\_mathematics\_behind\_balancing\_a\_game/](https://www.reddit.com/r/rpg/comments/yf1gbl/whats_the_mathematics_behind_balancing_a_game/)  
36. An Axiomatic Theory of Fairness in Resource Allocation \- Princeton ..., accessed November 7, 2025, [https://www.princeton.edu/\~chiangm/fairness.pdf](https://www.princeton.edu/~chiangm/fairness.pdf)  
37. accessed November 7, 2025, [https://geographicdata.science/book/notebooks/08\_point\_pattern\_analysis.html\#:\~:text=Thinking%20about%20these%20distributions%20of,probably%20have%20a%20clustered%20pattern.](https://geographicdata.science/book/notebooks/08_point_pattern_analysis.html#:~:text=Thinking%20about%20these%20distributions%20of,probably%20have%20a%20clustered%20pattern.)  
38. Point Pattern Analysis \- Geographic Data Science with Python, accessed November 7, 2025, [https://geographicdata.science/book/notebooks/08\_point\_pattern\_analysis.html](https://geographicdata.science/book/notebooks/08_point_pattern_analysis.html)  
39. Chapter 11 Analyzing Spatial Patterns | Intro to GIS and Spatial Analysis, accessed November 7, 2025, [https://mgimond.github.io/Spatial/chp11\_0.html](https://mgimond.github.io/Spatial/chp11_0.html)  
40. Moran's I and Getis-Ord G\* Analysis \- Matt Peeples, accessed November 7, 2025, [https://www.mattpeeples.net/modules/LISA.html](https://www.mattpeeples.net/modules/LISA.html)  
41. How High/Low Clustering (Getis-Ord General G) works—ArcGIS Pro | Documentation, accessed November 7, 2025, [https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-high-low-clustering-getis-ord-general-g-spat.htm](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-high-low-clustering-getis-ord-general-g-spat.htm)  
42. Module 7 Spatial Autocorrelation and Cluster Analysis \- RPubs, accessed November 7, 2025, [https://rpubs.com/Geesaale/1259986](https://rpubs.com/Geesaale/1259986)  
43. How Hot Spot Analysis (Getis-Ord Gi\*) works—ArcGIS Pro | Documentation, accessed November 7, 2025, [https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-hot-spot-analysis-getis-ord-gi-spatial-stati.htm](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-hot-spot-analysis-getis-ord-gi-spatial-stati.htm)  
44. How to calculate spatial hotspots and which tools do you need? \- CARTO, accessed November 7, 2025, [https://carto.com/blog/spatial-hotspot-tools](https://carto.com/blog/spatial-hotspot-tools)  
45. How Spatial Autocorrelation (Global Moran's I) works—ArcGIS Pro | Documentation, accessed November 7, 2025, [https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-spatial-autocorrelation-moran-s-i-spatial-st.htm](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-spatial-autocorrelation-moran-s-i-spatial-st.htm)  
46. Accessibility Via Public Transport Through Gravity Models Based on Open Data, accessed November 7, 2025, [https://www.researchgate.net/publication/386042107\_Accessibility\_Via\_Public\_Transport\_Through\_Gravity\_Models\_Based\_on\_Open\_Data](https://www.researchgate.net/publication/386042107_Accessibility_Via_Public_Transport_Through_Gravity_Models_Based_on_Open_Data)  
47. A Spatial Accessibility Study of Public Hospitals: A Multi-Mode Gravity-Based Two-Step Floating Catchment Area Method \- MDPI, accessed November 7, 2025, [https://www.mdpi.com/2076-3417/14/17/7713](https://www.mdpi.com/2076-3417/14/17/7713)  
48. Full article: Spatial Accessibility Patterns to Public Hospitals in Shanghai: An Improved Gravity Model \- Taylor & Francis Online, accessed November 7, 2025, [https://www.tandfonline.com/doi/full/10.1080/00330124.2021.2000445](https://www.tandfonline.com/doi/full/10.1080/00330124.2021.2000445)  
49. Gravity models for potential spatial healthcare access measurement: a systematic methodological review \- PMC \- PubMed Central, accessed November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10693160/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10693160/)  
50. Create and enrich grids and hexagons—ArcGIS Pro | Documentation, accessed November 7, 2025, [https://pro.arcgis.com/en/pro-app/3.3/help/analysis/business-analyst/generate-grids-and-hexagons.htm](https://pro.arcgis.com/en/pro-app/3.3/help/analysis/business-analyst/generate-grids-and-hexagons.htm)  
51. (PDF) Using hexagonal grids and network analysis for spatial accessibility assessment in urban environments \- A case study of public amenities in Toruń \- ResearchGate, accessed November 7, 2025, [https://www.researchgate.net/publication/330850358\_Using\_hexagonal\_grids\_and\_network\_analysis\_for\_spatial\_accessibility\_assessment\_in\_urban\_environments\_-\_A\_case\_study\_of\_public\_amenities\_in\_Torun](https://www.researchgate.net/publication/330850358_Using_hexagonal_grids_and_network_analysis_for_spatial_accessibility_assessment_in_urban_environments_-_A_case_study_of_public_amenities_in_Torun)  
52. A\* Pathfinding in a hexagonal grid \- Stack Overflow, accessed November 7, 2025, [https://stackoverflow.com/questions/38015645/a-pathfinding-in-a-hexagonal-grid](https://stackoverflow.com/questions/38015645/a-pathfinding-in-a-hexagonal-grid)  
53. Are square or hex grids better for pathfinding? \- Game Development Stack Exchange, accessed November 7, 2025, [https://gamedev.stackexchange.com/questions/3610/are-square-or-hex-grids-better-for-pathfinding](https://gamedev.stackexchange.com/questions/3610/are-square-or-hex-grids-better-for-pathfinding)  
54. (PDF) Map Generation and Balance in the Terra Mystica Board ..., accessed November 7, 2025, [https://www.researchgate.net/publication/342879278\_Map\_Generation\_and\_Balance\_in\_the\_Terra\_Mystica\_Board\_Game\_Using\_Particle\_Swarm\_and\_Local\_Search](https://www.researchgate.net/publication/342879278_Map_Generation_and_Balance_in_the_Terra_Mystica_Board_Game_Using_Particle_Swarm_and_Local_Search)  
55. Generating Map Sketches for Strategy Games \- Georgios N ..., accessed November 7, 2025, [https://yannakakis.net/wp-content/uploads/2013/08/liapis.pdf](https://yannakakis.net/wp-content/uploads/2013/08/liapis.pdf)  
56. Spatial Systems \- daniel.games | Board Game Design Articles, accessed November 7, 2025, [https://daniel.games/spatial-systems/](https://daniel.games/spatial-systems/)