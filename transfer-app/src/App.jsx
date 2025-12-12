import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import TransferTable from './components/TransferTable';
import { parseExcel, normalizeData, findMatches, generateBankFile } from './utils/excelProcessing';

function App() {
  const [providerData, setProviderData] = useState([]);
  const [appData, setAppData] = useState([]);
  const [matchedData, setMatchedData] = useState([]);
  const [uploadSteps, setUploadSteps] = useState({ provider: false, app: false });

  const handleProviderUpload = async (file) => {
    try {
      const data = await parseExcel(file);
      setProviderData(normalizeData(data));
      setUploadSteps(prev => ({ ...prev, provider: true }));
    } catch (error) {
      console.error("Error parsing provider file:", error);
    }
  };

  const handleAppUpload = async (file) => {
    try {
      const data = await parseExcel(file);
      setAppData(normalizeData(data));
      setUploadSteps(prev => ({ ...prev, app: true }));
    } catch (error) {
      console.error("Error parsing app file:", error);
    }
  };

  const handleProcessMatches = () => {
    if (providerData.length === 0 || appData.length === 0) return;

    const { matches } = findMatches(appData, providerData);
    setMatchedData(matches);

    if (matches.length === 0) {
      alert("No se encontraron coincidencias automáticas (buscando CUITs comunes).");
    }
  };

  const handleExport = () => {
    // We pass the currently matched (and selected handled by parent usually, but here we iterate matched)
    // Wait, TransferTable handles selection, but we need to pass the selection state up or method down
    // Actually TransferTable calls onSelectionChange, but we don't store it in parent matchedData exactly
    // Let's rely on TransferTable passing selection to a handler we store
    // Oh, wait, I need a state for selected items to export ONLY them.
  };

  const [selectedToExport, setSelectedToExport] = useState([]);

  return (
    <div className="min-h-screen bg-dark-950 p-6 md:p-12 text-white selection:bg-primary-500/30 font-sans">
      <div className="max-w-7xl mx-auto space-y-12">

        {/* Header */}
        <header className="text-center space-y-4 animate-in slide-in-from-top-8 duration-700 fade-in">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight bg-gradient-to-r from-white via-primary-200 to-primary-400 bg-clip-text text-transparent">
            Procesador de Transferencias
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Sube tus archivos para cruzar datos y generar el archivo de transferencias bancarias automáticamente.
          </p>
        </header>

        {/* Upload Section */}
        <div className="grid md:grid-cols-2 gap-8 animate-in slide-in-from-bottom-8 duration-700 fade-in delay-200">
          <div className="glass p-6 rounded-2xl space-y-4 transition-all duration-300 hover:bg-white/5 border border-white/5 hover:border-primary-500/30">
            <h2 className="text-xl font-semibold flex items-center gap-3">
              <span className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold transition-colors ${uploadSteps.provider ? 'bg-green-500/20 text-green-400' : 'bg-primary-500/20 text-primary-400'}`}>
                {uploadSteps.provider ? '✓' : '1'}
              </span>
              Base de Proveedores
            </h2>
            <p className="text-sm text-gray-400">
              Excel que contiene la base de datos maestra con CUITs y CBUs.
            </p>
            <FileUpload
              label="Arrastra o selecciona el archivo"
              onFileSelect={handleProviderUpload}
            />
          </div>

          <div className="glass p-6 rounded-2xl space-y-4 transition-all duration-300 hover:bg-white/5 border border-white/5 hover:border-primary-500/30">
            <h2 className="text-xl font-semibold flex items-center gap-3">
              <span className={`flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold transition-colors ${uploadSteps.app ? 'bg-green-500/20 text-green-400' : 'bg-primary-500/20 text-primary-400'}`}>
                {uploadSteps.app ? '✓' : '2'}
              </span>
              Registros de App
            </h2>
            <p className="text-sm text-gray-400">
              Excel exportado de la aplicación con los montos a transferir.
            </p>
            <FileUpload
              label="Arrastra o selecciona el archivo"
              onFileSelect={handleAppUpload}
            />
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-6 animate-in slide-in-from-bottom-8 duration-700 fade-in delay-300">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-semibold text-white">Resultados de Coincidencia</h2>
            {(uploadSteps.app && uploadSteps.provider) && (
              <button
                onClick={handleProcessMatches}
                className="px-4 py-2 rounded-lg bg-primary-500 hover:bg-primary-600 text-white font-medium transition-colors shadow-lg shadow-primary-500/20"
              >
                Procesar Coincidencias
              </button>
            )}
          </div>

          {matchedData.length > 0 ? (
            <TransferTable
              data={matchedData}
              onExport={() => generateBankFile(selectedToExport)}
              onSelectionChange={setSelectedToExport}
            />
          ) : (
            <div className="glass-card rounded-2xl p-12 flex flex-col items-center justify-center text-gray-500 border-dashed border-2 border-white/10">
              <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
                <TransferTableIcon className="text-gray-600" />
              </div>
              <p className="text-lg font-medium text-gray-400">Esperando archivos para procesar...</p>
              <p className="text-sm text-gray-600 mt-2">Sube ambos archivos arriba para comenzar el cruce de datos.</p>

              {/* Dev Only: Button to fill dummy data */}
              <button
                onClick={() => setMatchedData([
                  { id: 1, date: '2023-10-25', providerName: 'Tech Solutions SRL', cuit: '30-12345678-9', description: 'Servicios Octubre', amount: 150000.00, matchConfidence: 'exact', cbu: '000001230000' },
                  { id: 2, date: '2023-10-26', providerName: 'Librería El Estudiante', cuit: '30-87654321-0', description: 'Insumos varios', amount: 24500.50, matchConfidence: 'partial', cbu: '000004560000' },
                  { id: 3, date: '2023-10-27', providerName: 'Servicios de Limpieza SA', cuit: '33-11223344-5', description: 'Limpieza Mensual', amount: 89000.00, matchConfidence: 'exact', cbu: '000007890000' },
                ])}
                className="mt-8 text-xs text-primary-400/50 hover:text-primary-400 underline"
              >
                (Dev) Cargar Datos de Prueba
              </button>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

const TransferTableIcon = ({ className }) => (
  <svg className={className} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3v18" />
    <rect width="18" height="18" x="3" y="3" rx="2" />
    <path d="M3 9h18" />
    <path d="M3 15h18" />
  </svg>
);

export default App;
