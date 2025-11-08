/**
 * Balance Engine
 * Core logic for improving map balance using the ti4-map-lab algorithm
 */

import { tilesToBalanceMap, balanceMapToTiles } from './converters';
import { MAP_SPACE_TYPES } from './constants';

/**
 * Calculate home values for all home systems on the map
 * @param {Map} balanceMap - The map object from balance algorithm
 * @param {Object} evaluator - Evaluation variables
 * @returns {Array} Array of {space, value} objects
 */
function getHomeValues(balanceMap, evaluator) {
    const homeValues = [];

    for (let space of balanceMap.spaces) {
        if (space.type === MAP_SPACE_TYPES.HOME) {
            const value = balanceMap.getHomeValue(space, evaluator);
            homeValues.push({ space, value });
        }
    }

    return homeValues;
}

/**
 * Calculate the balance difference (max - min home values)
 * @param {Array} homeValues - Array of {space, value} objects
 * @returns {number} Balance gap
 */
function getBalanceDifference(homeValues) {
    if (homeValues.length === 0) return 0;

    const values = homeValues.map(hv => hv.value);
    const max = Math.max(...values);
    const min = Math.min(...values);

    return max - min;
}

/**
 * Check if a system can be swapped (not locked)
 * For now, we'll avoid moving:
 * - Wormhole systems
 * - Anomaly systems
 * - Empty systems
 *
 * @param {MapSpace} space - The space to check
 * @returns {boolean} True if the space can be swapped
 */
function canSwapSystem(space) {
    if (space.type !== MAP_SPACE_TYPES.SYSTEM) return false;
    if (!space.system) return false;

    // Don't swap systems with wormholes
    if (space.system.wormhole !== null) return false;

    // Don't swap systems with anomalies
    if (space.system.anomalies !== null) return false;

    // Don't swap systems with no planets (empty space)
    if (space.system.planets.length === 0) return false;

    // Don't swap Mecatol Rex
    if (space.system.isMecatolRexSystem()) return false;

    return true;
}

/**
 * Improve balance by swapping systems
 * This is based on the improveBalance() function from ti4-map-lab
 *
 * @param {Map} balanceMap - The map object
 * @param {Object} evaluator - Evaluation variables
 * @param {number} iterations - Number of swap attempts
 * @returns {number} Final balance gap
 */
export function improveBalance(balanceMap, evaluator, iterations = 100) {
    let homeValues = getHomeValues(balanceMap, evaluator);
    let balanceGap = getBalanceDifference(homeValues);

    // Get list of swappable systems
    const swappableSystems = balanceMap.spaces.filter(canSwapSystem);

    if (swappableSystems.length < 2) {
        console.warn('Not enough swappable systems for balancing');
        return balanceGap;
    }

    for (let i = 0; i < iterations; i++) {
        // Pick two random swappable systems
        const index1 = Math.floor(Math.random() * swappableSystems.length);
        let index2 = Math.floor(Math.random() * swappableSystems.length);

        // Make sure we don't pick the same system twice
        while (index2 === index1) {
            index2 = Math.floor(Math.random() * swappableSystems.length);
        }

        const space1 = swappableSystems[index1];
        const space2 = swappableSystems[index2];

        // Swap the systems
        const tempSystem = space1.system;
        space1.system = space2.system;
        space2.system = tempSystem;

        // Check if the balance improved
        const newHomeValues = getHomeValues(balanceMap, evaluator);
        const newBalanceGap = getBalanceDifference(newHomeValues);

        if (newBalanceGap < balanceGap) {
            // Accept the swap
            balanceGap = newBalanceGap;
            homeValues = newHomeValues;
        } else {
            // Revert the swap
            const tempSystem2 = space1.system;
            space1.system = space2.system;
            space2.system = tempSystem2;
        }
    }

    return balanceGap;
}

/**
 * Main balance function that takes ti4_map_generator data and returns balanced tiles
 *
 * @param {Array} tilesArray - Current tiles array from ti4_map_generator
 * @param {Object} tileData - All tile data
 * @param {Object} boardConfig - Board configuration
 * @param {Object} evaluator - Evaluator from default_evaluators.json
 * @param {number} iterations - Number of balance iterations
 * @returns {Object} {tiles: Array, balanceGap: number}
 */
export function balanceMap(tilesArray, tileData, boardConfig, evaluator, iterations = 100) {
    try {
        // Convert to balance map format
        const balanceMap = tilesToBalanceMap(tilesArray, tileData, boardConfig);

        // Run balance algorithm
        const balanceGap = improveBalance(balanceMap, evaluator, iterations);

        // Convert back to tiles array
        const balancedTiles = balanceMapToTiles(balanceMap, tilesArray.length);

        return {
            tiles: balancedTiles,
            balanceGap: balanceGap
        };
    } catch (error) {
        console.error('Error during map balancing:', error);
        throw error;
    }
}

/**
 * Calculate the current balance gap without making changes
 *
 * @param {Array} tilesArray - Current tiles array from ti4_map_generator
 * @param {Object} tileData - All tile data
 * @param {Object} boardConfig - Board configuration
 * @param {Object} evaluator - Evaluator from default_evaluators.json
 * @returns {number} Current balance gap
 */
export function calculateBalanceGap(tilesArray, tileData, boardConfig, evaluator) {
    try {
        const balanceMap = tilesToBalanceMap(tilesArray, tileData, boardConfig);
        const homeValues = getHomeValues(balanceMap, evaluator);
        return getBalanceDifference(homeValues);
    } catch (error) {
        console.error('Error calculating balance gap:', error);
        return null;
    }
}
