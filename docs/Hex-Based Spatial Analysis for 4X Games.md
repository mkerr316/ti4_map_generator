

# **Spatial Optimization in 4X Strategy Games: A Technical Analysis of Hexagonal Tiling, Algorithms, and AI**

## **I. The Strategic Choice of Tessellation: A Comparative Analysis of Grid Topologies**

The selection of a grid topology—the fundamental tessellation of the game world—represents the first and most critical act of spatial optimization in 4X game design. This decision, typically a choice between square and hexagonal tiles 1, pre-determines the spatial graph, constrains the efficacy of all subsequent algorithms, and dictates the core "feel" of unit movement and tactical interaction.2 While squares are often favored for their simple mapping to 2D arrays and player intuition with cardinal controls 2, a technical analysis reveals the hexagonal grid as a topologically superior framework for complex spatial reasoning.  
The primary deficiency of the square grid is its topological ambiguity, often termed the "diagonal problem".2 A square tile possesses two distinct classes of neighbors: four cardinal and four diagonal. The distance to cardinal neighbors is 1 unit, while the distance to diagonal neighbors is $\\sqrt{2}$ (approximately 1.414 units).2 This inconsistency forces developers into a non-optimal compromise: treating diagonal movement as 1 (making it disproportionately fast), treating it as 2 (making it prohibitively slow), or introducing floating-point movement costs, which are often undesirable.2 This ambiguity cascades into all spatial queries, complicating calculations for area-of-effect (AoE) and tactical positioning.6  
Hexagonal grids, by contrast, are defined by their topological uniformity.7 Each hexagon has six neighbors, and the distance from a tile's center to the center of all six adjacent tiles is identical.8 This singular neighbor relationship is not merely a convenience; it is a foundational algorithmic optimization. It eliminates the entire class of "diagonal" edge cases, simplifying all subsequent algorithms for pathfinding, line-of-sight, and influence propagation by removing the need to differentiate between two classes of adjacency.  
From a geospatial and analytical perspective, hexagons are demonstrably superior for representing continuous spatial data. As the most "circular" shape that can tessellate a plane, a hexagon features the lowest perimeter-to-area ratio, second only to a circle itself.12 This "circularity" is not an aesthetic concern but a mathematical one: it "reduce\[s\] sampling bias due to edge effects".13 When aggregating continuous data—such as "enemy threat" in an influence map or "resource value" in a procedural generator—a square grid's diagonals create "outlier" points at its corners.9 A hexagonal grid provides a "softer," more faithful discretization of this data and represents natural curves more effectively.9  
Tactically, the 8-neighbor property of squares is sometimes cited as offering more options.4 However, this creates topological anomalies; units diagonal to each other are "adjacent" for attacks but do not form a "connected" defensive line, allowing enemies to "leak" through. Hexagons, by contrast, create more robust tactical formations. A unit with two adjacent neighbors forms a "triangle" with them, a fundamentally more interconnected and stable local graph structure, making frontlines more comprehensible for both players and AI.14

| Metric | Hexagonal Grid | Square Grid |
| :---- | :---- | :---- |
| **Adjacency (Topology)** | Uniform: 6 neighbors of a single class. | Ambiguous: 8 neighbors of two classes (cardinal and diagonal).4 |
| **Neighbor Distance** | Consistent distance to all 6 neighbor centroids.8 | Inconsistent: Two distances (1 and $\\sqrt{2}$).2 |
| **Movement (Diagonal Problem)** | N/A. Movement cost is uniform in all directions. | A fundamental design challenge requiring "fudges" (costs 1, 1.5, or 2).2 |
| **Area of Effect (AoE) Bias** | Low bias. "More circular" shape provides a better approximation of a radius.13 | High bias. Corners are outliers, distorting AoE shapes.9 |
| **Tactical Formations** | Robust. Forms "triangles" and solid frontlines; chokepoints are unambiguous.14 | "Leaky." Gaps can exist between diagonally-adjacent units.4 |
| **Algorithmic Complexity** | Simplified. One class of neighbor simplifies all iterative algorithms. | Increased. Algorithms must constantly handle the "diagonal" edge case. |
| **Data Structure Mapping** | Less intuitive. Requires an abstract coordinate system (see Section II). | Highly intuitive. Maps directly to a 2D array.2 |
| **Player Intuition** | Less intuitive for cardinal controls; often seen as "geeky".2 | More intuitive; aligns with numpad and cardinal directions.4 |

**Table 1: Hex vs. Square Grid: A Comparative Analysis for 4X Game Design**

## **II. Algebraic Foundations and Representation of Hexagonal Grids**

The primary challenge of implementing a hexagonal grid is not graphical but algebraic. The most intuitive storage method, "offset coordinates" (such as odd-q or even-r), maps a hex grid to a 2D array by "shifting" every other row or column.15 While simple for storage, this system is algorithmically "clumsy".17 Critical operations such as distance calculation, neighbor finding, or rotation require complex, non-uniform logic. Standard vector operations (addition, subtraction, scaling) are not supported, preventing the use of elegant, generalized algorithms.18  
The canonical solution, popularized by the comprehensive guides from Red Blob Games, is to adopt an abstract algebraic system.18 This reframes the problem from "grid mapping" to "spatial algebra."

1. **Cube Coordinates:** This system represents the 2D hexagonal grid as a 2D plane sliced diagonally through a 3D Cartesian cube grid.18 Each hex is assigned three coordinates ($q$, $r$, $s$) which are constrained by the equation $q \+ r \+ s \= 0$.15 The power of this system is that it inherits the properties of a 3D vector space. Standard vector operations—addition, subtraction, scalar multiplication—work perfectly.18 This allows existing 3D Cartesian algorithms for distance, rotation, and line drawing to be adapted and reused.18  
2. **Axial Coordinates:** This system is "basically the same" as cube coordinates and is often preferred for implementation.15 It is a 2-axis system (e.g., $q$ and $r$) where the third coordinate is implicit, derived from the constraint: $s \= \-q \- r$.18 This provides the 2-coordinate simplicity needed for storage (e.g., as a key in a hash map) while retaining the full algebraic power of the 3-axis cube system for calculations.15

This abstraction is the key that unlocks all subsequent algorithmic simplicity. The developer is liberated from managing 2D array indices and if (row % 2 \== 0\) logic. The grid *becomes* the coordinate system, a continuous vector space that is merely sampled at integer ($q$, $r$, $s$) locations. This shift from ad-hoc offset "hacks" to a mathematically sound, unified model is the foundation of modern hex-grid development.

| Coordinate System | Storage | Algorithmic Simplicity | Vector Arithmetic | Distance Calculation |
| :---- | :---- | :---- | :---- | :---- |
| **Offset** (e.g., odd-q) | 2D Array (col, row) | Poor. Clumsy, requires many if statements and conversions.17 | Not supported. | Complex and slow. |
| **Axial** | 2-vector (q, r) | Excellent. | Supported (via conversion to Cube). | Simple (via conversion to Cube). |
| **Cube** | 3-vector (q, r, s) | Excellent. Most elegant for algorithms.15 | Natively supported (add, subtract, scale).18 | Simple: (abs(dq) \+ abs(dr) \+ abs(ds)) / 2\.18 |

**Table 2: Comparison of Hexagonal Coordinate Systems**

## **III. Core Spatial Algorithms for Hexagonal Environments**

Building upon the algebraic foundation of Cube and Axial coordinates, a suite of core spatial algorithms becomes trivial to implement.

* **Neighbors:** Finding all six adjacent tiles to a given hex $H$ is a simple vector addition. A set of six constant direction vectors (e.g., $N \= (0, 1, \-1)$, $NE \= (1, 0, \-1)$, etc.) is defined. Finding a neighbor is $H\_{neighbor} \= H\_{current} \+ \\text{direction}$.15  
* Distance: The "hex distance" (the number of steps in the shortest path) is derived directly from the Cube coordinates. For two hexes, $a$ and $b$, the distance is an adaptation of the 3D Manhattan distance, defined as:

  $$dist(a, b) \= \\frac{|a.q \- b.q| \+ |a.r \- b.r| \+ |a.s \- b.s|}{2}$$

  This single, elegant formula replaces all complex conditional logic required by offset systems.15  
* **Coordinate Conversion (Pixel-to-Hex):** A "Layout" class is necessary to manage the transformation between the abstract hex grid and the pixel-based screen.15 This class encapsulates the hex orientation (pointy-top or flat-top), size (allowing for stretched or squashed hexes), and origin (where hex $(0,0,0)$ is on the screen).15 Converting a pixel coordinate (e.g., from a mouse click) to a hex coordinate involves reversing the hex-to-pixel matrix transformation. This conversion yields *fractional* hex coordinates (e.g., $(1.2, \-0.8, \-0.4)$), which must then be "rounded" to the nearest valid integer hex coordinate. This rounding algorithm itself leverages the Cube coordinate system's properties to find the correct hex.15  
* **Line of Sight (LOS):** Calculating visibility is a more complex problem that reveals a fundamental tension in grid-based games. The literature describes two distinct methods for "line drawing" that are *not* interchangeable 21:  
  1. **Grid-Native Path (Modified Bresenham's):** This algorithm finds the "straightest" possible path composed *of hex tiles* from hex $A$ to hex $B$.21 This path respects the grid's topology and represents the shortest line a *unit can move along*.  
  2. **Euclidean Intersection (True LOS):** This algorithm determines which hexes intersect a *true Euclidean straight line* drawn from the center of $A$ to the center of $B$.21 This is the path a *bullet flies along* or the line a *character can see along*. The implementation involves checking if the line segment crosses the hex's boundaries by testing if any of its six vertices lie on the opposite side of the line from the others. This is typically done using a "turns" function, which is a 2D cross-product to determine if a point is left, right, or on the line.21

This duality creates an unavoidable conflict between *topological reality* (the grid path) and *geometric reality* (the visual line). A "straight" line of sight (Method 2\) might "clip" the corner of an obstacle hex that a "straight" path of movement (Method 1\) would have to navigate *around*. This can lead to player frustration, where a shot *looks* clear but is blocked by the grid logic, or a unit *moves* in a path that does not look perfectly straight. This "inconsistent" behavior is not a bug, but a non-obvious mathematical consequence of imposing a discrete grid on a continuous space.23

## **IV. Optimization of Movement: Pathfinding on Hexagonal Graphs**

In 4X games, unit movement is the most frequent spatial query. The *de facto* standard for this is the A\* search algorithm.24 The literature confirms that implementing A\* on a hex grid is "surprisingly easy".26 The A\* algorithm is generic; it works on *any* graph, and a hex grid is simply a graph where tiles are nodes and adjacencies are edges.2  
To adapt A\* for a hexagonal grid, only two grid-specific components are required 2:

1. **A Neighbor Function:** This provides the graph's edges. The hex\_neighbor function, derived from the Cube coordinate system (see Section III), fulfills this requirement perfectly.  
2. **A Heuristic Function ($H$):** This function estimates the cost from the current node to the target. For A\* to be *optimal* (guaranteed to find the shortest path), the heuristic must be *admissible*, meaning it *never* overestimates the true cost.

For a hex grid, the heuristic is a solved problem. The correct and perfectly admissible heuristic is the hex\_distance function (see Section III).2 This is the hex-grid equivalent of the "Manhattan distance" used as the heuristic for square grids.25  
The core A\* algorithm is, therefore, a solved problem. The *true* spatial optimization challenge for a 4X AI lies in the dynamic modeling of the **cost function ($G$)**, which represents the actual cost of moving from the start to the current node. The pathfinder is only as intelligent as the costs it is given.  
This $G$ cost is where all strategic depth is encoded. A\* natively supports weighted graphs, allowing it to pathfind around obstacles (impassable cost) or through difficult terrain (high cost).25 Academic literature proposes enhancing this with more sophisticated cost models, such as integrating "flexible setting of movement costs" based on arbitrary environmental data, or implementing "turning penalty" strategies.29  
This separates the *path-searcher* (A\*) from the *spatial-analyzer* (the cost model). A sophisticated 4X AI will not merely find the *shortest* path, but the *best* path. "Best" is a context-dependent, strategic term. For example:

* An *aggressive* AI will assign low $G$ costs to open terrain.  
* A *defensive* AI will assign low $G$ costs to tiles that offer a defensive bonus.  
* A *stealth* unit's AI will assign low $G$ costs to tiles with low enemy influence (see Section VI) and high costs to tiles within enemy LOS (see Section III).  
* A "turning penalty" 29 could be applied to a tank unit's $G$ cost for making a 120-degree turn, while an infantry unit would have no such penalty. This is how high-level strategic preferences are compiled down into the low-level spatial graph.

## **V. Modeling Territory: Spatial Partitioning and Control in 4X Systems**

Beyond local unit movement, 4X games require macro-scale spatial analysis for territory control and border definition. The literature presents a clear evolution of three primary methods.

### **Method 1: Static, Region-Based Systems**

This is a *game design* abstraction to solve spatial problems. The 4X game *Endless Legend* serves as a prime case study.30 The game map is pre-divided into "separate regions".7 The core mechanic is that when a player founds a city, "the *entire region* becomes part of a faction's territory".31 A hard constraint, "one city per region," is enforced.31 This design explicitly solves the "Infinite City Sprawl" (ICS) problem—where the optimal strategy is to "spam" as many cities as possible—that plagued earlier *Civilization* titles.32 While effective, this solution is rigid, and the region borders are static and predefined.

### **Method 2: Generative, Geometric Systems (Voronoi Diagrams)**

A more dynamic approach is to use Voronoi diagrams to partition the map.7 A Voronoi diagram divides a space into regions (cells) based on a set of "sites" (e.g., cities).34 Each cell contains all locations on the map that are *closest* to its corresponding site than to any other site.34 This "nearest neighbor" principle is a powerful and organic method for generating "AI territories" or "zones of control".33 It provides a dynamic model of cultural or political influence that expands naturally as players found new cities.

### **Method 3: Advanced, Weighted-Voronoi Systems**

The simple Voronoi model has a significant flaw: it is based on "as the crow flies" Euclidean distance. It does not account for impassable mountains, rivers, or the presence of a highway. A city across a mountain range might be geometrically "closest" but is functionally inaccessible.  
Academic research in geospatial analysis provides the next logical evolution: **Weighted Voronoi Diagrams**. One paper details a "Hexagon-based Adaptive Crystal Growth Voronoi Diagram".35 While its use case is delimiting public school service areas, its methodology is directly analogous to 4X territory control. This model is not based on pure distance, but is *weighted* by factors such as "accessibility (travel time based on the road network and natural barriers)" and "socioeconomic context (population)".35  
This "adaptive crystal growth" model is a blueprint for a highly sophisticated 4X territory system. Instead of borders being based on simple proximity, they would be based on a "travel-cost-Voronoi." In such a system, building roads would *actually* expand your effective border, and mountains would *actually* "push" it back, as territory would be defined by *accessibility* and *influence*, not just proximity.

## **VI. Influence Mapping for AI-Driven Spatial Optimization**

Influence Maps (IMs) are the primary data structure for AI spatial reasoning in strategy games.36 They are a "situation summary" that represents the state of the world in a numerical way that an AI can analyze.37 This allows an AI to quantify abstract concepts like "territorial control," "enemy threat," "frontlines," and "optimal placement".37 The literature reveals two competing architectures for this.

### **Architecture 1: Grid-Based Tactical Influence Maps (TIMs)**

This is a pre-computation, or *caching*, approach. The system maintains several "base maps" (e.g., EnemyThreatMap, AllyProximityMap) as 2D grids.38 When a unit moves, its influence is "stamped" onto these maps using a pre-calculated template that defines its falloff value.38  
The AI can then perform powerful queries by combining these maps. To find a safe location to retreat, the AI might query for the tile with the highest value in a "working map" generated by $AllyProximity \\times (1 \- EnemyThreat)$. To find the optimal location for an area-of-effect (AoE) spell, it would find the highest value in an $EnemyProximity$ map.38

* **Pro:** Queries are $O(1)$ (a simple array lookup).  
* **Con:** Updates are expensive ($O(m \\times k)$, where $m$ is units and $k$ is template size), as every unit's "stamp" must be added/removed from the grid every time it moves.

### **Architecture 2: Gridless "Infinite-Resolution" Influence Maps**

This is an *on-demand-computation*, or *querying*, approach.40 This method "escapes the grid" by *not* storing influence values in a grid at all. Instead, the system stores only a list of *influence sources* (points) and their associated *differentiable falloff functions* (e.g., $f(dist) \= 1 \- \\frac{dist}{radius}$).40  
To find the influence at *any* arbitrary point $(x,y)$, the AI iterates through all relevant sources, calculates their individual influence at $(x,y)$ using their falloff functions, and sums the results.40

* **Pro:** Updates are $O(1)$ (just update the source's position in a list). Memory cost is $O(m)$ (linear in the number of sources, $m$), not $O(width \\times height)$ for the grid.40  
* **Con:** Queries are expensive ($O(m)$), as every source must be checked.

This $O(m)$ query cost can be mitigated by using a spatial partitioning structure, such as a **k-d tree**, to quickly cull all sources whose radius of influence does not intersect the query point, bringing the average query cost down to $O(log m)$.40

### **Synthesis: A Hybrid 4X Solution**

The choice between these two architectures is a classic engineering trade-off. A 4X game AI has two distinct query profiles:

1. **High-Frequency, Tactical Queries:** "Where should this unit move *right now*?" This is needed for thousands of units, potentially many times per turn.  
2. **Low-Frequency, Strategic Queries:** "Where is the *globally* optimal hex on the entire map for my *next city*?".37 This is a high-importance, low-frequency query.

A hybrid architecture is the optimal solution. A *coarse, fast, grid-based IM* (Architecture 1\) should be used for all high-frequency tactical queries where $O(1)$ query speed is essential. Simultaneously, a *precise, slow, infinite-resolution IM* (Architecture 2\) should be used for high-importance, low-frequency strategic decisions like city placement, where precision (no grid discretization error) matters more than query speed.

## **VII. High-Performance Data Structures for Large-Scale Worlds**

The vast scale of 4X maps, potentially containing millions of tiles and thousands of active units, presents a significant systems engineering challenge. Efficiently storing and querying this spatial data is critical for performance.

### **Grid Storage: Array vs. Hash Map**

First, the map *itself* must be stored.

* **2D Array:** This is the most compact method, offering $O(1)$ access. However, it is only suitable for regular map shapes (rectangles, parallelograms) and requires "clumsy" index mapping (e.g., array\[r\]\[q \+ (r\>\>1)\] to convert from axial/cube coordinates).15  
* **Hash Table (e.g., std::unordered\_map):** This is the most flexible approach. By using the Axial Hex coordinate as the key, this method natively supports sparse maps, irregular shapes, and maps with holes.15

### **Object Storage: Cartesian Trees vs. Hexagonal Hierarchies**

Second, a spatial partitioning structure is required to store and query *objects on* the map, such as units, cities, and resources.41 A linear scan to answer "what objects are near this location?" 41 is computationally infeasible.  
The traditional solutions are **Quadtrees** and **K-d trees**.43 K-d trees are notoriously difficult to update dynamically as units move, often requiring a full rebuild.45 Quadtrees are more flexible but suffer from a fundamental "spatial impedance mismatch." A Quadtree is a *Cartesian* data structure; it works by recursively subdividing a *square* into four smaller *squares*.44 When a square-based Quadtree is forced on top of a *hexagonal* world, a single hex tile can straddle a Quadtree boundary. A leaf node in the tree will contain an awkward, non-uniform collection of hex fragments, leading to complex and inefficient range queries.  
The modern, state-of-the-art solution is a **Hexagonal Hierarchical Spatial Index**, such as Uber's open-source **H3** system.46 H3 is a *natively hexagonal* system. It partitions the world into large hexagons, each of which can be *recursively subdivided* into seven smaller hexagons (a problem that was notoriously difficult to solve geometrically 47). This structure combines the geometric advantages of a hex grid (uniform neighbor distance 8, low distortion 5) with the hierarchical $O(log n)$ query power of a Quadtree. For any large-scale 4X game built on a hex map (especially a spherical one, which H3 also supports), adopting an H3-like index for object querying is a profound optimization over force-fitting a Cartesian Quadtree.

| Data Structure | Primary Use | Storage Cost | Query Speed (kNN, Range) | Dynamic Update Cost | Suitability for Hex Grids |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **2D Array** | Storing the grid | $O(W \\times H)$ | $O(1)$ (for tile data) | Static | Poor. Only for regular map shapes.15 |
| **Hash Map** | Storing the grid | $O(N)$ (for $N$ tiles) | $O(1)$ (for tile data) | $O(1)$ | Excellent. Ideal for irregular/sparse maps.15 |
| **Quadtree** | Storing objects | $O(M \\log M)$ | $O(log M)$ | Moderate. Localized updates.45 | Poor. "Impedance mismatch".44 |
| **K-d Tree** | Storing objects | $O(M)$ | $O(log M)$ | High. Often requires full rebuild.45 | Poor. Cartesian structure. |
| **H3-like Index** | Storing objects | $O(M)$ | $O(log M)$ | Moderate. | Excellent. Natively hexagonal hierarchy.46 |

**Table 3: Spatial Query Data Structures: A Performance and Use-Case Matrix** (W=Width, H=Height, N=NumTiles, M=NumObjects)

## **VIII. Procedural Generation and the Problem of "Balance"**

In 4X games, Procedural Content Generation (PCG) is essential for replayability, ensuring each new game offers a diverse and engaging environment.48 Standard techniques involve layering noise functions (e.g., Perlin, FastNoiseLite) to generate heightmaps and terrain types 50, and using Voronoi diagrams to delineate continents.7  
However, 4X games have a higher-order requirement than simple variety: **balance**. Generating a map "purely at random" is insufficient, as it can lead to a "non-functional game" 54—for example, by giving one player all the critical resources and another player none. The central optimization problem of 4X PCG is ensuring *fairness* and "strategically interesting" layouts.55  
The literature points to a solution that reframes PCG from a simple *constructive* process into a *search-based* optimization problem.

1. **Generate:** A constructive algorithm (e.g., noise-based) generates a candidate map.  
2. **Evaluate:** A high-level **evaluation function** (also called a "reward function" 57 or "fitness function") "scores" the generated map based on a set of balance and quality metrics.  
3. **Optimize:** This score is used to drive an optimization algorithm (like an evolutionary algorithm or simple "generate-and-test") that searches for a map that *maximizes* the evaluation score.55

The challenge is then shifted from *generating* a perfect map to *writing* an evaluation function that can programmatically quantify abstract concepts like "balanced start," "fair resource distribution," and "good choke points".56 This moves the spatial optimization problem from the in-game AI to the content creation pipeline itself.

## **IX. Emerging Frontiers: Graph-Based Map Analysis and Optimization**

The evaluation functions described in Section VIII are notoriously difficult to write by hand.58 How does one write a static function that captures the *dynamic* and *contextual* value of a tile?  
The emerging state-of-the-art solution is to *learn* this evaluation function using Graph Neural Networks (GNNs). This approach models the 4X map not as a grid, but as a formal **graph**.58

* **Nodes:** Each hex tile (or planet in a stellar 4X) is a node.58  
* **Edges:** Adjacency links (or starlanes) form the edges.  
* **Features:** Each node is assigned a feature vector representing its properties, such as \[terrain\_type, resource\_value, has\_city, owner\_faction, influence\_value\_N\].61

The GNN's task is to learn the *implicit value* of a node based on both its own features and its topological position within the graph. A specific GNN architecture, **GraphSAGE** (Graph Sample and Aggregation), is identified as uniquely suited for this task.58  
GraphSAGE works by generating a new, "learned" representation (an embedding) for a node by sampling the feature vectors from its local neighborhood and *aggregating* them.67 Its single most important property is that it is an **inductive** learning framework.66

* *Transductive* models (like standard Graph Convolutional Networks) require the *entire* graph to be present during training. They cannot generalize to new, unseen graphs.  
* *Inductive* models, like GraphSAGE, learn a *function* for generating embeddings. This function can then be applied to "efficiently generate representations on previously unseen data".66

This inductive property is the key. A developer can train a GraphSAGE model on data from thousands of simulated games.58 The model will *learn* the contextual value of map positions. It will learn, for example, that a node with high\_resource features and low\_connectivity (a one-neighbor chokepoint) is a "high-value safe zone," while a node with low\_resource features but high\_connectivity (high betweenness-centrality) is a "critical strategic chokepoint."  
A study applying this exact method to a stellar 4X game successfully trained a GraphSAGE model to perform "planet classification" with 94% accuracy.58 This trained, inductive GNN *is* the next-generation evaluation function. It can be plugged directly into the PCG pipeline from Section VIII to provide a sophisticated, data-driven "balance score" for any newly generated, unseen map, far surpassing the capabilities of any hand-written heuristic.

## **X. Synthesis: Design Philosophy and Systemic Integration**

This analysis has detailed a hierarchy of spatial optimization, from low-level algebraic representations to high-level machine learning models. However, an analysis of 4X game design postmortems reveals a final, "meta-optimization": the *dissolution* of spatial problems through game design.  
4X games are fundamentally "efficiency engine" games.70 The core loop is "ramping up production" 70, which leads to a "snowball" effect where an early spatial advantage (e.g., optimal city placement) becomes an unstoppable lead.71 This makes 4X AI an "really complex issue" 72, as the AI must manage "multiple conflicting priorities" 72 across a massive spatial and temporal state space 73—a task at which it notoriously fails.74  
The postmortem for *Old World*, by *Civilization IV* lead designer Soren Johnson, provides a powerful counter-point.32 The design faced two classic spatial optimization problems:

1. **"Infinite City Sprawl" (ICS):** The purely spatial problem of "where to place cities" led to an unfun "city spam" strategy.32  
2. **"Traffic Jams":** The "one unit per tile" (1UPT) rule created spatial logjams and uninteresting unit-shuffling.32

The solutions were *not* more complex spatial algorithms. They were *game design abstractions* that re-contextualized the problem.

* The solution to ICS was not a GNN-powered city-placement AI (Section IX). The solution was *design*: **"preset city sites"**.32 This *dissolved* the spatial "where" problem and replaced it with a temporal "when" and "if" problem.  
* The solution to traffic jams was not a complex, multi-agent pathfinding algorithm. The solution was *design*: the **"Orders" system**.32 "Orders" are a scarce, global resource required for *any* unit action. Because unit movement is now prohibitively expensive, players (and AI) *cannot* move every unit every turn. This *non-spatial* economic constraint solves the *spatial* logjam problem.

This reveals the ultimate trade-off in 4X design and optimization. A developer can either solve a spatial problem by **simulating** it with increasing algorithmic complexity (e.g., Weighted Voronoi territory, GNN-based evaluations) or by **abstracting** it away with a clever, hard-coded game mechanic. The first path leads to more emergent, complex, and simulation-ist systems. The second leads to more constrained, balanced, and "board game-like" systems. The choice between these two philosophies is the highest-level optimization decision a 4X designer must make.

#### **Works cited**

1. Hexagon vs normal square grid : r/gamedesign \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/gamedesign/comments/1cnf1yr/hexagon\_vs\_normal\_square\_grid/](https://www.reddit.com/r/gamedesign/comments/1cnf1yr/hexagon_vs_normal_square_grid/)  
2. Are square or hex grids better for pathfinding? \- Game Development Stack Exchange, accessed November 7, 2025, [https://gamedev.stackexchange.com/questions/3610/are-square-or-hex-grids-better-for-pathfinding](https://gamedev.stackexchange.com/questions/3610/are-square-or-hex-grids-better-for-pathfinding)  
3. Let's talk about the use of a grid or hexes in games and how they affect things. \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/truegaming/comments/28zvch/lets\_talk\_about\_the\_use\_of\_a\_grid\_or\_hexes\_in/](https://www.reddit.com/r/truegaming/comments/28zvch/lets_talk_about_the_use_of_a_grid_or_hexes_in/)  
4. Squares vs hexes in 4X strategy games \- a question as old as time : r/gamedesign \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/gamedesign/comments/pg1tlc/squares\_vs\_hexes\_in\_4x\_strategy\_games\_a\_question/](https://www.reddit.com/r/gamedesign/comments/pg1tlc/squares_vs_hexes_in_4x_strategy_games_a_question/)  
5. accessed November 7, 2025, [https://medium.com/@altair0622/spatial-joins-at-scale-the-hidden-trade-offs-behind-h3-and-geometry-3d88e703bb02\#:\~:text=You%20can%20use%20square%20grids,No%20natural%20hierarchical%20relationships](https://medium.com/@altair0622/spatial-joins-at-scale-the-hidden-trade-offs-behind-h3-and-geometry-3d88e703bb02#:~:text=You%20can%20use%20square%20grids,No%20natural%20hierarchical%20relationships)  
6. Hexes v Squares \- Luke Savage \- WordPress.com, accessed November 7, 2025, [https://lukebsavage.wordpress.com/2015/04/13/hexes-v-squares/](https://lukebsavage.wordpress.com/2015/04/13/hexes-v-squares/)  
7. Any 4X games with interesting map topologies (non-square, non-hexagon)? \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/4Xgaming/comments/1brqo2r/any\_4x\_games\_with\_interesting\_map\_topologies/](https://www.reddit.com/r/4Xgaming/comments/1brqo2r/any_4x_games_with_interesting_map_topologies/)  
8. H3 Hexagonal Grid \- Kontur Inc., accessed November 7, 2025, [https://www.kontur.io/blog/why-we-use-h3/](https://www.kontur.io/blog/why-we-use-h3/)  
9. Hexagons for Location Intelligence: Why, When & How? \- CARTO, accessed November 7, 2025, [https://carto.com/blog/hexagons-for-location-intelligence](https://carto.com/blog/hexagons-for-location-intelligence)  
10. Hex map \- Wikipedia, accessed November 7, 2025, [https://en.wikipedia.org/wiki/Hex\_map](https://en.wikipedia.org/wiki/Hex_map)  
11. Which has more advantages, a hex or square tile for turn-based strategy games? \- Quora, accessed November 7, 2025, [https://www.quora.com/Which-has-more-advantages-a-hex-or-square-tile-for-turn-based-strategy-games](https://www.quora.com/Which-has-more-advantages-a-hex-or-square-tile-for-turn-based-strategy-games)  
12. Fishnets, Honeycombs and Footballs; Better spatial models with hexagonal grids \- Medium, accessed November 7, 2025, [https://medium.com/@goldrydigital/fishnets-honeycombs-and-footballs-better-spatial-models-with-hexagonal-grids-768bdf92d3bb](https://medium.com/@goldrydigital/fishnets-honeycombs-and-footballs-better-spatial-models-with-hexagonal-grids-768bdf92d3bb)  
13. Why hexagons?—ArcGIS Pro | Documentation, accessed November 7, 2025, [https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-whyhexagons.htm](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-whyhexagons.htm)  
14. Hex vs Square \- Horrible Pain, accessed November 7, 2025, [http://horriblepain.com/2016/03/hex-vs-square/](http://horriblepain.com/2016/03/hex-vs-square/)  
15. Implementation of Hex Grids \- Red Blob Games, accessed November 7, 2025, [https://www.redblobgames.com/grids/hexagons/implementation.html](https://www.redblobgames.com/grids/hexagons/implementation.html)  
16. Hexagon Grids \- multiple ways to approach it. Which system to use? : r/gamedev \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/gamedev/comments/54qbac/hexagon\_grids\_multiple\_ways\_to\_approach\_it\_which/](https://www.reddit.com/r/gamedev/comments/54qbac/hexagon_grids_multiple_ways_to_approach_it_which/)  
17. Amit Patel's Guide for Hex Grids and Gamelogic Grids, accessed November 7, 2025, [https://gamelogic.co.za/documentation/grids2/content/AmitPatel.html](https://gamelogic.co.za/documentation/grids2/content/AmitPatel.html)  
18. Hexagonal Grids \- Red Blob Games, accessed November 7, 2025, [https://www.redblobgames.com/grids/hexagons/](https://www.redblobgames.com/grids/hexagons/)  
19. Hexagonal grids \- A Lazy Sequence, accessed November 7, 2025, [https://brehaut.net/blog/2021/hexagonal\_grids/](https://brehaut.net/blog/2021/hexagonal_grids/)  
20. data structures \- How do I represent a hextile/hex grid in memory? \- Stack Overflow, accessed November 7, 2025, [https://stackoverflow.com/questions/1838656/how-do-i-represent-a-hextile-hex-grid-in-memory](https://stackoverflow.com/questions/1838656/how-do-i-represent-a-hextile-hex-grid-in-memory)  
21. Clark Verbrugge's Hex Grids \- Stanford, accessed November 7, 2025, [http://www-cs-students.stanford.edu/\~amitp/Articles/HexLOS.html](http://www-cs-students.stanford.edu/~amitp/Articles/HexLOS.html)  
22. Line of sight in a hex grid : r/gamedev \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/gamedev/comments/s5qs4/line\_of\_sight\_in\_a\_hex\_grid/](https://www.reddit.com/r/gamedev/comments/s5qs4/line_of_sight_in_a_hex_grid/)  
23. Inconsistent movement / line-of-sight around obstacles on a hexagonal grid, accessed November 7, 2025, [https://gamedev.stackexchange.com/questions/77464/inconsistent-movement-line-of-sight-around-obstacles-on-a-hexagonal-grid](https://gamedev.stackexchange.com/questions/77464/inconsistent-movement-line-of-sight-around-obstacles-on-a-hexagonal-grid)  
24. Red Blob Games, accessed November 7, 2025, [https://www.redblobgames.com/](https://www.redblobgames.com/)  
25. Pathfinding on a hexagonal grid \- A\* Algorithm \- The Knights of U, accessed November 7, 2025, [https://theknightsofu.com/pathfinding-on-a-hexagonal-grid-a-algorithm-2/](https://theknightsofu.com/pathfinding-on-a-hexagonal-grid-a-algorithm-2/)  
26. Unity Pathfinding on a Hex Grid System\! \- YouTube, accessed November 7, 2025, [https://www.youtube.com/watch?v=XqEBu3un1ik](https://www.youtube.com/watch?v=XqEBu3un1ik)  
27. Pathfinding on an hexagonal grid using A\* : r/gamedev \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/gamedev/comments/1lzpz5q/pathfinding\_on\_an\_hexagonal\_grid\_using\_a/](https://www.reddit.com/r/gamedev/comments/1lzpz5q/pathfinding_on_an_hexagonal_grid_using_a/)  
28. Finding shortest path on a hexagonal grid \- Game Development Stack Exchange, accessed November 7, 2025, [https://gamedev.stackexchange.com/questions/44812/finding-shortest-path-on-a-hexagonal-grid](https://gamedev.stackexchange.com/questions/44812/finding-shortest-path-on-a-hexagonal-grid)  
29. Improved A\* Navigation Path-Planning Algorithm Based on Hexagonal Grid \- MDPI, accessed November 7, 2025, [https://www.mdpi.com/2220-9964/13/5/166](https://www.mdpi.com/2220-9964/13/5/166)  
30. Endless Legend Preview \- SpaceSector.com, accessed November 7, 2025, [https://www.spacesector.com/blog/2014/08/endless-legend-preview/](https://www.spacesector.com/blog/2014/08/endless-legend-preview/)  
31. Endless Legend \- Wikipedia, accessed November 7, 2025, [https://en.wikipedia.org/wiki/Endless\_Legend](https://en.wikipedia.org/wiki/Endless_Legend)  
32. My Elephant in the Room, Part 1 | DESIGNER NOTES, accessed November 7, 2025, [http://www.designer-notes.com/my-elephant-in-the-room-part-1/](http://www.designer-notes.com/my-elephant-in-the-room-part-1/)  
33. Generating "territories" on a hexagonal grid : r/gamedev \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/gamedev/comments/2cmbig/generating\_territories\_on\_a\_hexagonal\_grid/](https://www.reddit.com/r/gamedev/comments/2cmbig/generating_territories_on_a_hexagonal_grid/)  
34. Voronoi Diagrams in Game Development — Procedural Maps, AI ..., accessed November 7, 2025, [https://www.gamegeniuslab.com/tutorial-post/voronoi-diagrams-in-game-development-procedural-maps-ai-territories-stylish-effects/](https://www.gamegeniuslab.com/tutorial-post/voronoi-diagrams-in-game-development-procedural-maps-ai-territories-stylish-effects/)  
35. Hexagon-Based Adaptive Crystal Growth Voronoi Diagrams Based ..., accessed November 7, 2025, [https://www.mdpi.com/2220-9964/7/7/257](https://www.mdpi.com/2220-9964/7/7/257)  
36. Influence Map-Based Pathfinding Algorithms in Video Games \- uBibliorum, accessed November 7, 2025, [https://ubibliorum.ubi.pt/bitstream/10400.6/5517/1/3443\_6881.pdf](https://ubibliorum.ubi.pt/bitstream/10400.6/5517/1/3443_6881.pdf)  
37. Influence Maps \- Colin Deane, accessed November 7, 2025, [https://www.colindeane.me/influence-maps.html](https://www.colindeane.me/influence-maps.html)  
38. Modular Tactical Influence Maps \- Game AI Pro, accessed November 7, 2025, [https://www.gameaipro.com/GameAIPro2/GameAIPro2\_Chapter30\_Modular\_Tactical\_Influence\_Maps.pdf](https://www.gameaipro.com/GameAIPro2/GameAIPro2_Chapter30_Modular_Tactical_Influence_Maps.pdf)  
39. Classic Game Postmortem: Sid Meier's Civilization \- YouTube, accessed November 7, 2025, [https://www.youtube.com/watch?v=AJ-auWfJTts](https://www.youtube.com/watch?v=AJ-auWfJTts)  
40. Escaping the Grid: Infinite-Resolution Influence ... \- Game AI Pro, accessed November 7, 2025, [http://www.gameaipro.com/GameAIPro2/GameAIPro2\_Chapter29\_Escaping\_the\_Grid\_Infinite-Resolution\_Influence\_Mapping.pdf](http://www.gameaipro.com/GameAIPro2/GameAIPro2_Chapter29_Escaping_the_Grid_Infinite-Resolution_Influence_Mapping.pdf)  
41. Spatial Partition \- Game Programming Patterns, accessed November 7, 2025, [https://gameprogrammingpatterns.com/spatial-partition.html](https://gameprogrammingpatterns.com/spatial-partition.html)  
42. Geospatial | GeoHash | R-Tree | QuadTree | by Tarun Jain \- Medium, accessed November 7, 2025, [https://tarunjain07.medium.com/geospatial-geohash-notes-15cbc50b329d](https://tarunjain07.medium.com/geospatial-geohash-notes-15cbc50b329d)  
43. amay12/SpatialSearch: This project compares and analyzes the performance of two popular spatial indexing data structures: K-D Trees and Quadtrees for insertion, search and finding the nearest neighbors of points on a 2D plane. \- GitHub, accessed November 7, 2025, [https://github.com/amay12/SpatialSearch](https://github.com/amay12/SpatialSearch)  
44. Gym, QuadTree, and k-d tree \- Andriy Buday, accessed November 7, 2025, [https://andriybuday.com/2020/01/gym-quadtree-and-k-d-tree.html](https://andriybuday.com/2020/01/gym-quadtree-and-k-d-tree.html)  
45. Fully dynamic KD-Tree vs. Quadtree? \- Game Development Stack Exchange, accessed November 7, 2025, [https://gamedev.stackexchange.com/questions/87138/fully-dynamic-kd-tree-vs-quadtree](https://gamedev.stackexchange.com/questions/87138/fully-dynamic-kd-tree-vs-quadtree)  
46. Guide to Uber's H3 for Spatial Indexing \- Analytics Vidhya, accessed November 7, 2025, [https://www.analyticsvidhya.com/blog/2025/03/ubers-h3-for-spatial-indexing/](https://www.analyticsvidhya.com/blog/2025/03/ubers-h3-for-spatial-indexing/)  
47. Hierarchical Hexagonal Clustering and Indexing \- MDPI, accessed November 7, 2025, [https://www.mdpi.com/2073-8994/11/6/731](https://www.mdpi.com/2073-8994/11/6/731)  
48. Procedural Content Generation for video games, a friendly approach, accessed November 7, 2025, [https://www.levelup-gamedevhub.com/en/news/procedural-content-generation-for-video-games-a-friendly-approach/](https://www.levelup-gamedevhub.com/en/news/procedural-content-generation-for-video-games-a-friendly-approach/)  
49. Creating a Newer and Improved Procedural Content Generation (PCG) Algorithm with Minimal Human Intervention for Computer Gaming Development \- MDPI, accessed November 7, 2025, [https://www.mdpi.com/2073-431X/13/11/304](https://www.mdpi.com/2073-431X/13/11/304)  
50. GODOT PROCEDURAL GENERATION \- Hexagonal Maps (C\#) \- YouTube, accessed November 7, 2025, [https://www.youtube.com/watch?v=djM8EZG3Dz0](https://www.youtube.com/watch?v=djM8EZG3Dz0)  
51. Progress on map generation for my hex based strategy game : r/Unity3D \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/Unity3D/comments/1jlljm6/progress\_on\_map\_generation\_for\_my\_hex\_based/](https://www.reddit.com/r/Unity3D/comments/1jlljm6/progress_on_map_generation_for_my_hex_based/)  
52. Procedurally Generating A Hexagon Grid in Unity \- YouTube, accessed November 7, 2025, [https://www.youtube.com/watch?v=EPaSmQ2vtek](https://www.youtube.com/watch?v=EPaSmQ2vtek)  
53. I created a hex based procedural map generator to try and teach myself Godot programming. Feel free to play around with it. The larger map sizes can take a little while to generate, as the efficiency isn't very good yet, particularly with river creation. : r/proceduralgeneration \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/proceduralgeneration/comments/1ak8pr6/i\_created\_a\_hex\_based\_procedural\_map\_generator\_to/](https://www.reddit.com/r/proceduralgeneration/comments/1ak8pr6/i_created_a_hex_based_procedural_map_generator_to/)  
54. Procedural Content Generation via Generative Artificial Intelligence \- arXiv, accessed November 7, 2025, [https://arxiv.org/html/2407.09013v1](https://arxiv.org/html/2407.09013v1)  
55. Bachelor Thesis Project Generation, evaluation, and optimisation of procedural 2D tile-based maps in turn-based tactical video games \- DiVA portal, accessed November 7, 2025, [http://www.diva-portal.org/smash/get/diva2:945531/FULLTEXT01.pdf](http://www.diva-portal.org/smash/get/diva2:945531/FULLTEXT01.pdf)  
56. What are the top 3 most important things for you to enjoy a 4x? : r/4Xgaming \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/4Xgaming/comments/1jmtc5w/what\_are\_the\_top\_3\_most\_important\_things\_for\_you/](https://www.reddit.com/r/4Xgaming/comments/1jmtc5w/what_are_the_top_3_most_important_things_for_you/)  
57. arxiv.org, accessed November 7, 2025, [https://arxiv.org/html/2501.08552v1](https://arxiv.org/html/2501.08552v1)  
58. BALANCING MECHANISM IN STELLAR 4X GAMES USING GRAPHSAGE-BASED INDUCTIVE REPRESENTATION LEARNING | Request PDF \- ResearchGate, accessed November 7, 2025, [https://www.researchgate.net/publication/389324736\_BALANCING\_MECHANISM\_IN\_STELLAR\_4X\_GAMES\_USING\_GRAPHSAGE-BASED\_INDUCTIVE\_REPRESENTATION\_LEARNING](https://www.researchgate.net/publication/389324736_BALANCING_MECHANISM_IN_STELLAR_4X_GAMES_USING_GRAPHSAGE-BASED_INDUCTIVE_REPRESENTATION_LEARNING)  
59. (PDF) What is Game Balancing? An Examination of Concepts \- ResearchGate, accessed November 7, 2025, [https://www.researchgate.net/publication/340849177\_What\_is\_Game\_Balancing\_An\_Examination\_of\_Concepts](https://www.researchgate.net/publication/340849177_What_is_Game_Balancing_An_Examination_of_Concepts)  
60. Coding GraphSAGE From Scratch \- Syed A. Rizvi, accessed November 7, 2025, [https://syedarizvi.com/blog/2024/graphsage-from-scratch/](https://syedarizvi.com/blog/2024/graphsage-from-scratch/)  
61. GNNExplainer: Generating Explanations for Graph Neural Networks \- Stanford Computer Science, accessed November 7, 2025, [https://cs.stanford.edu/people/jure/pubs/gnnexplainer-neurips19.pdf](https://cs.stanford.edu/people/jure/pubs/gnnexplainer-neurips19.pdf)  
62. HEX-GNN: Hierarchical EXpanders for Node Classification | Request PDF \- ResearchGate, accessed November 7, 2025, [https://www.researchgate.net/publication/381368585\_HEX-GNN\_Hierarchical\_EXpanders\_for\_Node\_Classification](https://www.researchgate.net/publication/381368585_HEX-GNN_Hierarchical_EXpanders_for_Node_Classification)  
63. Applying Graph Neural Networks on Heterogeneous Nodes and Edge Features, accessed November 7, 2025, [https://grlearning.github.io/papers/6.pdf](https://grlearning.github.io/papers/6.pdf)  
64. Interpretability of Graph Neural Networks: An Exploratory Study of Nodes, Features, and Attention | by Francesco De Bernardis | The Quantastic Journal | Medium, accessed November 7, 2025, [https://medium.com/the-quantastic-journal/interpretability-of-graph-neural-networks-an-exploratory-study-of-nodes-features-and-attention-70799aec74dd](https://medium.com/the-quantastic-journal/interpretability-of-graph-neural-networks-an-exploratory-study-of-nodes-features-and-attention-70799aec74dd)  
65. accessed November 7, 2025, [https://snap.stanford.edu/graphsage/\#:\~:text=GraphSAGE%20is%20a%20framework%20for,have%20rich%20node%20attribute%20information.](https://snap.stanford.edu/graphsage/#:~:text=GraphSAGE%20is%20a%20framework%20for,have%20rich%20node%20attribute%20information.)  
66. GraphSAGE: Inductive Representation Learning on Large Graphs \- Stanford University, accessed November 7, 2025, [https://snap.stanford.edu/graphsage/](https://snap.stanford.edu/graphsage/)  
67. What is GraphSAGE? | Activeloop Glossary, accessed November 7, 2025, [https://www.activeloop.ai/resources/glossary/graph-sage/](https://www.activeloop.ai/resources/glossary/graph-sage/)  
68. OhMyGraphs: GraphSAGE and inductive representation learning | by Nabila Abraham | Analytics Vidhya | Medium, accessed November 7, 2025, [https://medium.com/analytics-vidhya/ohmygraphs-graphsage-and-inductive-representation-learning-ea26d2835331](https://medium.com/analytics-vidhya/ohmygraphs-graphsage-and-inductive-representation-learning-ea26d2835331)  
69. Modeling Game State Using Graph Neural Networks (GNNs) | by Pauline Arnoud | Stanford CS224W \- Medium, accessed November 7, 2025, [https://medium.com/stanford-cs224w/modeling-game-state-using-graph-neural-networks-gnns-ead8f790e54c](https://medium.com/stanford-cs224w/modeling-game-state-using-graph-neural-networks-gnns-ead8f790e54c)  
70. The Curious Case of 4X Games, Efficiency Engines, and Missing Strategic Gambits, accessed November 7, 2025, [http://www.big-game-theory.com/2020/01/the-curious-case-of-4x-games-efficiency.html](http://www.big-game-theory.com/2020/01/the-curious-case-of-4x-games-efficiency.html)  
71. Growth prevention mechanics in 4X games \- how to make them more interesting? \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/gamedesign/comments/10muqjc/growth\_prevention\_mechanics\_in\_4x\_games\_how\_to/](https://www.reddit.com/r/gamedesign/comments/10muqjc/growth_prevention_mechanics_in_4x_games_how_to/)  
72. AI for global decision-making in 4X games \- Game Development Stack Exchange, accessed November 7, 2025, [https://gamedev.stackexchange.com/questions/206216/ai-for-global-decision-making-in-4x-games](https://gamedev.stackexchange.com/questions/206216/ai-for-global-decision-making-in-4x-games)  
73. AI Design in 4X Games – An Approach \- SpaceSector.com, accessed November 7, 2025, [https://www.spacesector.com/blog/2012/12/ai-design-in-4x-games-an-overview/](https://www.spacesector.com/blog/2012/12/ai-design-in-4x-games-an-overview/)  
74. AI for 4X games \- Expectations, and the Way Forward. : r/4Xgaming \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/4Xgaming/comments/8ws2o2/ai\_for\_4x\_games\_expectations\_and\_the\_way\_forward/](https://www.reddit.com/r/4Xgaming/comments/8ws2o2/ai_for_4x_games_expectations_and_the_way_forward/)  
75. Feedback: the AI is braindead and passive... again : r/EndlessLegend \- Reddit, accessed November 7, 2025, [https://www.reddit.com/r/EndlessLegend/comments/1myjupw/feedback\_the\_ai\_is\_braindead\_and\_passive\_again/](https://www.reddit.com/r/EndlessLegend/comments/1myjupw/feedback_the_ai_is_braindead_and_passive_again/)  
76. My Elephant in the Room: An 'Old World' Postmortem \- YouTube, accessed November 7, 2025, [https://www.youtube.com/watch?v=VXH9\_d\_pDa4](https://www.youtube.com/watch?v=VXH9_d_pDa4)