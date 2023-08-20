const { start } = require("./dist/src/sim");

function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function asyncFunction() {
    const games = start(1);
    await delay(2000); // Waits for 2 seconds
    const state = games[0].p1.getState();
    console.log(`State Bits: `, state.length / 2);
}

asyncFunction();
