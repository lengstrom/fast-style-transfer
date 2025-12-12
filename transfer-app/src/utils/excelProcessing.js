import { read, utils, write } from 'xlsx';

export const parseExcel = async (file) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const data = e.target?.result;
                const workbook = read(data, { type: 'array' });

                // Priority: 'AGENDA' sheet (Provider DB), otherwise first sheet
                let sheetName = workbook.SheetNames[0];
                if (workbook.SheetNames.includes('AGENDA')) {
                    sheetName = 'AGENDA';
                    console.log("Found AGENDA sheet, using it.");
                }

                const sheet = workbook.Sheets[sheetName];

                // Convert to array of arrays first to find the header row
                const rawRows = utils.sheet_to_json(sheet, { header: 1 });

                // Find header row: look for row containing 'cuit' or 'razsocial' or 'nombre'
                let headerRowIndex = 0;
                const knownHeaders = ['cuit', 'razsocial', 'importe', 'detalle', 'cbu', 'nombre'];

                for (let i = 0; i < Math.min(rawRows.length, 10); i++) {
                    const rowStr = JSON.stringify(rawRows[i]).toLowerCase();
                    const matchCount = knownHeaders.filter(h => rowStr.includes(h)).length;
                    // Lower threshold to 1 if we find 'cuit' specifically, as it is unique
                    if (matchCount >= 2 || (rowStr.includes('cuit') && rowStr.includes('nombre'))) {
                        headerRowIndex = i;
                        break;
                    }
                }

                // Now parse again using this header row
                const jsonData = utils.sheet_to_json(sheet, { range: headerRowIndex });
                resolve(jsonData);
            } catch (error) {
                reject(error);
            }
        };

        reader.onerror = (error) => reject(error);
        reader.readAsArrayBuffer(file);
    });
};

/**
 * Normalizes keys to standard internal format
 */
export const normalizeData = (data) => {
    return data.map(row => {
        const newRow = {};
        Object.keys(row).forEach(key => {
            const normalizedKey = key.toLowerCase().trim().replace(/\s+/g, '_');

            // Map specific Spanish columns to internal keys
            if (normalizedKey === 'razsocial' || normalizedKey === 'beneficiario') newRow['providerName'] = row[key];
            else if (normalizedKey === 'haber') newRow['amount'] = row[key]; // Screenshot confirms 'haber' has the values
            else if (normalizedKey === 'detalle') newRow['filterDetail'] = row[key]; // Keep for D.Directo DD check
            else if (normalizedKey === 'coment') newRow['description'] = row[key]; // Screenshot 'coment' has the rich detail
            else if (normalizedKey === 'cuit' || normalizedKey === 'cuil') newRow['cuit'] = String(row[key]).replace(/[^0-9]/g, '');
            else if (normalizedKey === 'cbu') newRow['cbu'] = row[key];
            else if (normalizedKey === 'fecha' || normalizedKey === 'fecval') newRow['date'] = row[key];

            newRow[normalizedKey] = row[key];
        });

        // Fill defaults if missing from mapping but present in raw normalized keys
        if (!newRow.providerName && newRow.nombre) newRow.providerName = newRow.nombre;
        // Final fallback logic for amount
        if (!newRow.amount && newRow.monto) newRow.amount = newRow.monto;
        if (!newRow.amount && newRow.importe) newRow.amount = newRow.importe; // Fallback if Haber is missing


        return { ...row, ...newRow };
    });
};

/**
 * Helper to extract clean CUIT from a row
 */
const getCuit = (row) => {
    // Priority keys
    const keys = Object.keys(row);
    const cuitKey = keys.find(k =>
        ['cuit', 'cuil', 'nro_doc', 'identificacion', 'tax_id'].some(term => k.includes(term))
    );

    let val = cuitKey ? row[cuitKey] : null;

    // Fallback: look for value that looks like CUIT (11 digits) if no key match
    if (!val) {
        val = Object.values(row).find(v => {
            const str = String(v).replace(/[^0-9]/g, '');
            return str.length === 11;
        });
    }

    return val ? String(val).replace(/[^0-9]/g, '') : null;
};

export const findMatches = (appRecords, providerDb) => {
    const providerMap = new Map();

    // Index providers by CUIT (normalized)
    providerDb.forEach(row => {
        const cuit = getCuit(row);
        if (cuit) providerMap.set(cuit, row);
    });

    const matches = [];
    const unmatched = [];

    // Special CUITs list from VBA script
    const specialCuits = new Set([
        "20271370440", "20280864588", "20304161923",
        "20264164371", "20313523862", "20180310046"
    ]);

    appRecords.forEach((record, index) => {
        // 1. Filter logic: 'detalle' must be 'D.Directo DD'
        // We mapped 'detalle' to 'filterDetail' in normalizeData
        const detalleVal = record['filterDetail'] || record['detalle'];
        if (String(detalleVal).trim() !== "D.Directo DD") {
            return; // Skip this record
        }

        const cuit = getCuit(record);

        // EXCLUDE special CUITs (Anticipos Choferes) as requested
        if (cuit && specialCuits.has(cuit)) {
            return;
        }

        const cuitVal = String(cuit || "");

        // Attempt standard match
        if (cuit && providerMap.has(cuit)) {
            const provider = providerMap.get(cuit);

            const providerName = provider.nombre || provider.razon_social || provider.proveedor || 'Proveedor Encontrado';

            // Exclude specific providers by name as requested
            if (providerName.toLowerCase().includes("carganet")) {
                return;
            }

            const cbu = provider.cbu || provider.cuenta || '';
            const email = provider['Email destinatario'] || provider.email || '';

            // User says "coment" is the detail. We mapped 'coment' -> 'description'.
            const realDetail = record['description'] || record['coment'] || 'Varios';

            // Exclude records that look like Retentions if not needed
            if (String(realDetail).toLowerCase().includes("retenci")) {
                return;
            }

            const amount = record.amount || 0;
            const date = record.date || new Date().toLocaleDateString();

            matches.push({
                id: `match-${index}`,
                originalId: index,
                cuit,
                providerName,
                cbu,
                description: realDetail,
                email,
                amount,
                date,
                status: 'ready',
                matchConfidence: 'exact',
                originalRecord: record,
                providerRecord: provider
            });
        } else {
            const realDetail = record['description'] || record['coment'] || 'Varios';

            // Exclude Carganet and Retentions here too just in case
            if (String(realDetail).toLowerCase().includes("retenci")) return;

            unmatched.push({
                id: `unmatched-${index}`,
                cuit,
                providerName: 'Proveedor Desconocido',
                cbu: '',
                description: realDetail,
                amount: record.amount || 0,
                date: record.date,
                matchConfidence: 'none'
            });
        }
    });

    return { matches, unmatched };
};

export const generateBankFile = (selectedRecords) => {
    // Headers based on typical bank requirements (can be customized)
    const dataToExport = selectedRecords.map(rec => ({
        'CBU Destino': rec.cbu,
        'CUIT Destino': rec.cuit,
        'Importe': rec.amount,
        'Referencia': rec.description,
        'Fecha': rec.date,
        'Beneficiario': rec.providerName
    }));

    const ws = utils.json_to_sheet(dataToExport);
    const wb = utils.book_new();
    utils.book_append_sheet(wb, ws, "Transferencias");

    // Generate download
    const wbout = write(wb, { bookType: 'xlsx', type: 'array' });
    const blob = new Blob([wbout], { type: 'application/octet-stream' });

    // Trigger download
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `transferencias_banco_${new Date().toISOString().slice(0, 10)}.xlsx`;
    a.click();
    window.URL.revokeObjectURL(url);
};
