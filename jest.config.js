module.exports = {
    transform: {
        "^.+\\.ts?$": "ts-jest",
    },
    testRegex: "(/tests/.*|(\\.|/)(test|spec))\\.(js?|ts?)$",
    testPathIgnorePatterns: ["/dist/", "/node_modules/", "/env/"],
    moduleFileExtensions: ["ts", "tsx", "js", "jsx", "json", "node"],
    collectCoverage: true,
};
