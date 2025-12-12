import pkg from 'xlsx';
const { readFile, utils } = pkg;

const filePath = "C:\\Users\\Hernán\\Documents\\TRONADOR APP\\TRANSFER\\DB TRANSFER.xlsm";
const targetSheet = "AGENDA";

try {
    console.log("Attempting to read:", filePath);
    const workbook = readFile(filePath);

    if (!workbook.SheetNames.includes(targetSheet)) {
        console.log(`Sheet "${targetSheet}" not found. Available sheets:`, workbook.SheetNames);
    } else {
        console.log(`Sheet "${targetSheet}" found.`);
        const sheet = workbook.Sheets[targetSheet];

        // Get headers (first few rows to scan for structure)
        const rows = utils.sheet_to_json(sheet, { header: 1 });

        console.log("--- PROVIDER DB INSPECTION ---");
        // Print first 5 rows to identify where headers likely are
        rows.slice(0, 5).forEach((row, index) => {
            console.log(`Row ${index}:`, row);
        });
        console.log("------------------------");
    }

} catch (error) {
    console.error("Error reading file:", error.message);
    console.error(error);
}
