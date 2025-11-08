/**
 * Data converters between ti4_map_generator and ti4-map-lab balance algorithm
 *
 * ti4_map_generator uses strings for traits, specialties, anomalies, wormholes
 * ti4-map-lab balance algorithm uses numbers for these values
 */

import {
    PLANET_TRAITS as GEN_PLANET_TRAITS,
    TECH_SPECIALTIES as GEN_TECH_SPECIALTIES,
    ANOMALIES as GEN_ANOMALIES,
    WORMHOLES as GEN_WORMHOLES
} from '../data/tileData';

import {
    PLANET_TRAITS as BAL_PLANET_TRAITS,
    TECH_SPECIALTIES as BAL_TECH_SPECIALTIES,
    ANOMALIES as BAL_ANOMALIES,
    WORMHOLES as BAL_WORMHOLES,
    MAP_SPACE_TYPES
} from './constants';

import { PlanetBox, SystemBox, Map, MapSpace } from './map-logic';

// ========================================
// String to Number Converters (for balance algorithm input)
// ========================================

/**
 * Convert string planet trait to balance algorithm number
 */
export function traitStringToNumber(traitString) {
    if (traitString === null || traitString === undefined) return null;

    switch (traitString) {
        case GEN_PLANET_TRAITS.HAZARDOUS:
            return BAL_PLANET_TRAITS.HAZARDOUS;
        case GEN_PLANET_TRAITS.INDUSTRIAL:
            return BAL_PLANET_TRAITS.INDUSTRIAL;
        case GEN_PLANET_TRAITS.CULTURAL:
            return BAL_PLANET_TRAITS.CULTURAL;
        default:
            return null;
    }
}

/**
 * Convert string tech specialty to balance algorithm number
 */
export function techSpecialtyStringToNumber(techString) {
    if (techString === null || techString === undefined) return null;

    switch (techString) {
        case GEN_TECH_SPECIALTIES.BIOTIC:
            return BAL_TECH_SPECIALTIES.BIOTIC;
        case GEN_TECH_SPECIALTIES.WARFARE:
            return BAL_TECH_SPECIALTIES.WARFARE;
        case GEN_TECH_SPECIALTIES.PROPULSION:
            return BAL_TECH_SPECIALTIES.PROPULSION;
        case GEN_TECH_SPECIALTIES.CYBERNETIC:
            return BAL_TECH_SPECIALTIES.CYBERNETIC;
        default:
            return null;
    }
}

/**
 * Convert string anomaly to balance algorithm number
 */
export function anomalyStringToNumber(anomalyString) {
    if (anomalyString === null || anomalyString === undefined) return null;

    switch (anomalyString) {
        case GEN_ANOMALIES.NEBULA:
            return BAL_ANOMALIES.NEBULA;
        case GEN_ANOMALIES.GRAVITY_RIFT:
            return BAL_ANOMALIES.GRAVITY_RIFT;
        case GEN_ANOMALIES.ASTEROID_FIELD:
            return BAL_ANOMALIES.ASTEROID_FIELD;
        case GEN_ANOMALIES.SUPERNOVA:
            return BAL_ANOMALIES.SUPERNOVA;
        // Note: ENTROPIC_SCAR only exists in balance algorithm
        default:
            return null;
    }
}

/**
 * Convert string wormhole to balance algorithm number
 */
export function wormholeStringToNumber(wormholeString) {
    if (wormholeString === null || wormholeString === undefined) return null;

    switch (wormholeString) {
        case GEN_WORMHOLES.ALPHA:
            return BAL_WORMHOLES.ALPHA;
        case GEN_WORMHOLES.BETA:
            return BAL_WORMHOLES.BETA;
        case GEN_WORMHOLES.DELTA:
            return BAL_WORMHOLES.DELTA;
        // Note: Not all wormholes from generator are in balance algorithm
        default:
            return null;
    }
}

// ========================================
// Number to String Converters (for balance algorithm output)
// ========================================

/**
 * Convert balance algorithm number to string planet trait
 */
export function traitNumberToString(traitNumber) {
    if (traitNumber === null || traitNumber === undefined) return null;

    switch (traitNumber) {
        case BAL_PLANET_TRAITS.HAZARDOUS:
            return GEN_PLANET_TRAITS.HAZARDOUS;
        case BAL_PLANET_TRAITS.INDUSTRIAL:
            return GEN_PLANET_TRAITS.INDUSTRIAL;
        case BAL_PLANET_TRAITS.CULTURAL:
            return GEN_PLANET_TRAITS.CULTURAL;
        default:
            return null;
    }
}

/**
 * Convert balance algorithm number to string tech specialty
 */
export function techSpecialtyNumberToString(techNumber) {
    if (techNumber === null || techNumber === undefined) return null;

    switch (techNumber) {
        case BAL_TECH_SPECIALTIES.BIOTIC:
            return GEN_TECH_SPECIALTIES.BIOTIC;
        case BAL_TECH_SPECIALTIES.WARFARE:
            return GEN_TECH_SPECIALTIES.WARFARE;
        case BAL_TECH_SPECIALTIES.PROPULSION:
            return GEN_TECH_SPECIALTIES.PROPULSION;
        case BAL_TECH_SPECIALTIES.CYBERNETIC:
            return GEN_TECH_SPECIALTIES.CYBERNETIC;
        default:
            return null;
    }
}

/**
 * Convert balance algorithm number to string anomaly
 */
export function anomalyNumberToString(anomalyNumber) {
    if (anomalyNumber === null || anomalyNumber === undefined) return null;

    switch (anomalyNumber) {
        case BAL_ANOMALIES.NEBULA:
            return GEN_ANOMALIES.NEBULA;
        case BAL_ANOMALIES.GRAVITY_RIFT:
            return GEN_ANOMALIES.GRAVITY_RIFT;
        case BAL_ANOMALIES.ASTEROID_FIELD:
            return GEN_ANOMALIES.ASTEROID_FIELD;
        case BAL_ANOMALIES.SUPERNOVA:
            return GEN_ANOMALIES.SUPERNOVA;
        default:
            return null;
    }
}

/**
 * Convert balance algorithm number to string wormhole
 */
export function wormholeNumberToString(wormholeNumber) {
    if (wormholeNumber === null || wormholeNumber === undefined) return null;

    switch (wormholeNumber) {
        case BAL_WORMHOLES.ALPHA:
            return GEN_WORMHOLES.ALPHA;
        case BAL_WORMHOLES.BETA:
            return GEN_WORMHOLES.BETA;
        case BAL_WORMHOLES.DELTA:
            return GEN_WORMHOLES.DELTA;
        default:
            return null;
    }
}

// ========================================
// Coordinate System Converters
// ========================================

/**
 * Cube coordinate system helpers
 * ti4-map-lab uses cube coordinates (x, y, z) where x + y + z = 0
 * ti4_map_generator uses flat array indices
 *
 * The standard TI4 map is arranged in a hex grid with cube coordinates:
 * Center (Mecatol Rex) is at (0, 0, 0)
 */

/**
 * Convert flat array index to cube coordinates
 * This requires knowing the board layout structure
 */
export function indexToCubeCoords(index, playerCount, mapStyle = 'normal') {
    // This is a simplified version - may need adjustment based on actual board layouts
    // The exact mapping depends on the specific board configuration

    // For now, we'll use a basic hexagonal arrangement
    // Center tile (usually Mecatol) is at index 19 for 6-player standard

    // Standard 6-player map centered at index 19
    const standardCoordMap = {
        0: { x: 0, y: 3, z: -3 },
        1: { x: 1, y: 2, z: -3 },
        2: { x: 2, y: 1, z: -3 },
        3: { x: 3, y: 0, z: -3 },
        4: { x: -1, y: 3, z: -2 },
        5: { x: 0, y: 2, z: -2 },
        6: { x: 1, y: 1, z: -2 },
        7: { x: 2, y: 0, z: -2 },
        8: { x: 3, y: -1, z: -2 },
        9: { x: -2, y: 3, z: -1 },
        10: { x: -1, y: 2, z: -1 },
        11: { x: 0, y: 1, z: -1 },
        12: { x: 1, y: 0, z: -1 },
        13: { x: 2, y: -1, z: -1 },
        14: { x: 3, y: -2, z: -1 },
        15: { x: -3, y: 3, z: 0 },
        16: { x: -2, y: 2, z: 0 },
        17: { x: -1, y: 1, z: 0 },
        18: { x: 0, y: 0, z: 0 },    // Center (usually Mecatol Rex)
        19: { x: 1, y: -1, z: 0 },
        20: { x: 2, y: -2, z: 0 },
        21: { x: 3, y: -3, z: 0 },
        22: { x: -3, y: 2, z: 1 },
        23: { x: -2, y: 1, z: 1 },
        24: { x: -1, y: 0, z: 1 },
        25: { x: 0, y: -1, z: 1 },
        26: { x: 1, y: -2, z: 1 },
        27: { x: 2, y: -3, z: 1 },
        28: { x: -3, y: 1, z: 2 },
        29: { x: -2, y: 0, z: 2 },
        30: { x: -1, y: -1, z: 2 },
        31: { x: 0, y: -2, z: 2 },
        32: { x: 1, y: -3, z: 2 },
        33: { x: -3, y: 0, z: 3 },
        34: { x: -2, y: -1, z: 3 },
        35: { x: -1, y: -2, z: 3 },
        36: { x: 0, y: -3, z: 3 },
    };

    return standardCoordMap[index] || { x: 0, y: 0, z: 0 };
}

/**
 * Convert cube coordinates to flat array index
 */
export function cubeCoordsToIndex(coords) {
    // Reverse lookup of the coordinate map
    // This is inefficient but works for now
    const standardCoordMap = {
        '0,3,-3': 0,
        '1,2,-3': 1,
        '2,1,-3': 2,
        '3,0,-3': 3,
        '-1,3,-2': 4,
        '0,2,-2': 5,
        '1,1,-2': 6,
        '2,0,-2': 7,
        '3,-1,-2': 8,
        '-2,3,-1': 9,
        '-1,2,-1': 10,
        '0,1,-1': 11,
        '1,0,-1': 12,
        '2,-1,-1': 13,
        '3,-2,-1': 14,
        '-3,3,0': 15,
        '-2,2,0': 16,
        '-1,1,0': 17,
        '0,0,0': 18,
        '1,-1,0': 19,
        '2,-2,0': 20,
        '3,-3,0': 21,
        '-3,2,1': 22,
        '-2,1,1': 23,
        '-1,0,1': 24,
        '0,-1,1': 25,
        '1,-2,1': 26,
        '2,-3,1': 27,
        '-3,1,2': 28,
        '-2,0,2': 29,
        '-1,-1,2': 30,
        '0,-2,2': 31,
        '1,-3,2': 32,
        '-3,0,3': 33,
        '-2,-1,3': 34,
        '-1,-2,3': 35,
        '0,-3,3': 36,
    };

    const key = `${coords.x},${coords.y},${coords.z}`;
    return standardCoordMap[key] !== undefined ? standardCoordMap[key] : -1;
}

// ========================================
// Main Conversion Functions
// ========================================

/**
 * Convert ti4_map_generator tile data to balance algorithm system data
 */
export function tileDataToSystemData(tileData) {
    const systemData = {
        id: parseInt(tileData.id) || 0,
        anomalies: null,
        wormhole: null,
        planets: []
    };

    // Convert anomalies
    if (tileData.anomaly && tileData.anomaly.length > 0) {
        systemData.anomalies = tileData.anomaly
            .map(a => anomalyStringToNumber(a))
            .filter(a => a !== null);
        if (systemData.anomalies.length === 0) systemData.anomalies = null;
    }

    // Convert wormhole (only first one if multiple)
    if (tileData.wormhole && tileData.wormhole.length > 0) {
        systemData.wormhole = wormholeStringToNumber(tileData.wormhole[0]);
    }

    // Convert planets
    if (tileData.planets && tileData.planets.length > 0) {
        systemData.planets = tileData.planets.map(planet => planet.name);
    }

    return systemData;
}

/**
 * Convert ti4_map_generator planet data to balance algorithm planet data
 */
export function planetDataToPlanetData(planetData) {
    const balancePlanetData = {
        name: planetData.name,
        resources: planetData.resources || 0,
        influence: planetData.influence || 0,
        traits: null,
        tech_specialties: null
    };

    // Convert trait
    if (planetData.trait) {
        const trait = traitStringToNumber(planetData.trait);
        if (trait !== null) {
            balancePlanetData.traits = [trait];
        }
    }

    // Convert tech specialty
    if (planetData.specialty) {
        const specialty = techSpecialtyStringToNumber(planetData.specialty);
        if (specialty !== null) {
            balancePlanetData.tech_specialties = [specialty];
        }
    }

    return balancePlanetData;
}

/**
 * Convert ti4_map_generator tiles array to balance algorithm Map object
 * @param {Array} tilesArray - Array of tile IDs from ti4_map_generator
 * @param {Object} tileData - All tile data from tileData.js
 * @param {Object} boardConfig - Board configuration (home_worlds, etc.)
 * @returns {Map} Balance algorithm Map object
 */
export function tilesToBalanceMap(tilesArray, tileData, boardConfig) {
    // Create planet box with all planet data
    const allPlanetData = [];
    Object.values(tileData.all).forEach(tile => {
        if (tile.planets) {
            tile.planets.forEach(planet => {
                allPlanetData.push(planetDataToPlanetData(planet));
            });
        }
    });
    const planetBox = new PlanetBox(allPlanetData);

    // Create system box with all system data
    const allSystemData = Object.entries(tileData.all).map(([id, tile]) => {
        const systemData = tileDataToSystemData({ ...tile, id });
        return systemData;
    });
    const systemBox = new SystemBox(allSystemData, planetBox);

    // Create map with spaces
    const balanceMap = new Map(false);

    // Convert each tile in the array to a MapSpace
    tilesArray.forEach((tileId, index) => {
        if (tileId === -1 || tileId === null) return; // Skip empty tiles

        const coords = indexToCubeCoords(index);
        const tile = tileData.all[tileId];

        if (!tile) return;

        let spaceType = MAP_SPACE_TYPES.SYSTEM;
        let system = systemBox.getSystemByID(parseInt(tileId));

        // Determine if this is a home world
        if (boardConfig.home_worlds && boardConfig.home_worlds.includes(index)) {
            spaceType = MAP_SPACE_TYPES.HOME;
        }

        const mapSpace = new MapSpace(
            coords.x,
            coords.y,
            coords.z,
            null, // warp_spaces
            spaceType,
            system
        );

        balanceMap.spaces.push(mapSpace);
    });

    return balanceMap;
}

/**
 * Convert balance algorithm Map object back to ti4_map_generator tiles array
 * @param {Map} balanceMap - Balance algorithm Map object
 * @param {number} arraySize - Size of the tiles array (37 or 61)
 * @returns {Array} Array of tile IDs
 */
export function balanceMapToTiles(balanceMap, arraySize = 37) {
    const tilesArray = new Array(arraySize).fill(-1);

    balanceMap.spaces.forEach(space => {
        if (space.system) {
            const index = cubeCoordsToIndex(space);
            if (index !== -1) {
                tilesArray[index] = space.system.id;
            }
        }
    });

    return tilesArray;
}
