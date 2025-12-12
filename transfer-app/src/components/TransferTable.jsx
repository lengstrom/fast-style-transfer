import React, { useMemo, useState } from 'react';
import { Download, CheckSquare, Square, ArrowRight, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
    return twMerge(clsx(inputs));
}

const TransferTable = ({ data = [], onExport, onSelectionChange }) => {
    const [selectedIds, setSelectedIds] = useState(new Set());
    const [lastSelectedId, setLastSelectedId] = useState(null);

    // Mock data structure expectation handling
    // Expected data: { id, date, amount, description, providerName, cbu, cuit, status, matchConfidence }

    const handleSelectAll = () => {
        if (selectedIds.size === data.length) {
            setSelectedIds(new Set());
            onSelectionChange([]);
        } else {
            const allIds = new Set(data.map(item => item.id));
            setSelectedIds(allIds);
            onSelectionChange(data);
        }
    };

    const handleSelectOne = (id, multiSelect = false) => {
        const newSelected = new Set(selectedIds);
        if (newSelected.has(id)) {
            newSelected.delete(id);
        } else {
            newSelected.add(id);
        }
        setSelectedIds(newSelected);
        setLastSelectedId(id);

        const selectedItems = data.filter(item => newSelected.has(item.id));
        onSelectionChange(selectedItems);
    };

    const totalAmount = useMemo(() => {
        return data
            .filter(item => selectedIds.has(item.id))
            .reduce((acc, curr) => acc + (Number(curr.amount) || 0), 0);
    }, [selectedIds, data]);

    if (data.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center p-12 text-gray-400">
                <div className="bg-dark-800/50 p-4 rounded-full mb-4">
                    <AlertCircle size={32} />
                </div>
                <p>No hay datos coincidientes para mostrar.</p>
            </div>
        );
    }

    return (
        <div className="w-full space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">

            {/* Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-4 glass p-4 rounded-xl">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 text-sm text-gray-300">
                        <span className="font-semibold text-white">{selectedIds.size}</span>
                        seleccionados
                    </div>
                    <div className="h-4 w-px bg-white/10" />
                    <div className="flex items-center gap-2 text-sm text-gray-300">
                        Total:
                        <span className="font-semibold text-primary-400">
                            {new Intl.NumberFormat('es-AR', { style: 'currency', currency: 'ARS' }).format(totalAmount)}
                        </span>
                    </div>
                </div>

                <button
                    onClick={onExport}
                    disabled={selectedIds.size === 0}
                    className={cn(
                        "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all duration-300",
                        selectedIds.size > 0
                            ? "bg-primary-500 hover:bg-primary-600 text-white shadow-lg shadow-primary-500/20"
                            : "bg-dark-800 text-gray-500 cursor-not-allowed"
                    )}
                >
                    <Download size={18} />
                    Exportar Bank File
                </button>
            </div>

            {/* Table Container */}
            <div className="glass-card rounded-xl overflow-hidden border border-white/5">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="text-xs text-gray-400 uppercase bg-black/20">
                            <tr>
                                <th scope="col" className="p-4 w-4">
                                    <div className="flex items-center">
                                        <button
                                            onClick={handleSelectAll}
                                            className="text-gray-400 hover:text-white transition-colors"
                                        >
                                            {selectedIds.size === data.length && data.length > 0 ? (
                                                <CheckSquare size={18} className="text-primary-400" />
                                            ) : (
                                                <Square size={18} />
                                            )}
                                        </button>
                                    </div>
                                </th>
                                <th scope="col" className="px-6 py-3">Fecha</th>
                                <th scope="col" className="px-6 py-3">Proveedor</th>
                                <th scope="col" className="px-6 py-3">CUIT</th>
                                <th scope="col" className="px-6 py-3">Detalle</th>
                                <th scope="col" className="px-6 py-3 text-right">Monto</th>
                                <th scope="col" className="px-6 py-3 text-center">Estado</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                            {data.map((row) => (
                                <tr
                                    key={row.id}
                                    className={cn(
                                        "hover:bg-white/5 transition-colors cursor-pointer",
                                        selectedIds.has(row.id) ? "bg-primary-500/5 hover:bg-primary-500/10" : ""
                                    )}
                                    onClick={() => handleSelectOne(row.id)}
                                >
                                    <td className="w-4 p-4">
                                        <div className="flex items-center">
                                            {selectedIds.has(row.id) ? (
                                                <CheckSquare size={18} className="text-primary-400" />
                                            ) : (
                                                <Square size={18} className="text-gray-600" />
                                            )}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-gray-300 whitespace-nowrap">{row.date}</td>
                                    <td className="px-6 py-4 font-medium text-white whitespace-nowrap">{row.providerName}</td>
                                    <td className="px-6 py-4 text-gray-400 font-mono text-xs">{row.cuit}</td>
                                    <td className="px-6 py-4 text-gray-400 max-w-xs truncate">{row.description}</td>
                                    <td className="px-6 py-4 text-right font-medium text-white whitespace-nowrap">
                                        {new Intl.NumberFormat('es-AR', { style: 'currency', currency: 'ARS' }).format(row.amount)}
                                    </td>
                                    <td className="px-6 py-4 text-center">
                                        <span className={cn(
                                            "px-2 py-1 rounded-full text-xs font-medium border",
                                            row.matchConfidence === 'exact'
                                                ? "bg-green-500/10 text-green-400 border-green-500/20"
                                                : "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
                                        )}>
                                            {row.matchConfidence === 'exact' ? 'Exacto' : 'Parcial'}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default TransferTable;
