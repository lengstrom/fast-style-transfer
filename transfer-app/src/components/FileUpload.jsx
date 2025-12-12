import React, { useCallback, useState } from 'react';
import { Upload, FileSpreadsheet, CheckCircle, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
    return twMerge(clsx(inputs));
}

const FileUpload = ({ label, onFileSelect, acceptedFileTypes = ".xlsx, .xls, .xlsm" }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [fileName, setFileName] = useState(null);
    const [error, setError] = useState(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (file) => {
        // Validar extensiones permitidas
        const validExtensions = ['.xlsx', '.xls', '.xlsm'];
        const isExtensionValid = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

        if (!isExtensionValid) {
            setError('Por favor, sube un archivo Excel válido (.xlsx, .xls, .xlsm)');
            setFileName(null);
            return;
        }
        setError(null);
        setFileName(file.name);
        onFileSelect(file);
    };

    return (
        <div className="w-full">
            <label className="block text-sm font-medium text-gray-300 mb-2 font-sans">
                {label}
            </label>
            <div
                className={cn(
                    "relative group cursor-pointer transition-all duration-300 ease-in-out",
                    "border-2 border-dashed rounded-xl p-8",
                    "flex flex-col items-center justify-center gap-4",
                    "glass-card hover:bg-white/10",
                    isDragging ? "border-primary-500 bg-white/5 scale-[1.02]" : "border-white/10 hover:border-primary-400/50",
                    fileName ? "border-green-500/50" : ""
                )}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    accept={acceptedFileTypes}
                    onChange={handleChange}
                />

                <div className="flex flex-col items-center text-center gap-2 pointer-events-none">
                    {fileName ? (
                        <>
                            <div className="p-3 rounded-full bg-green-500/20 text-green-400 animate-in zoom-in duration-300">
                                <CheckCircle size={32} />
                            </div>
                            <div>
                                <p className="text-white font-semibold">{fileName}</p>
                                <p className="text-sm text-green-400">Archivo listo para procesar</p>
                            </div>
                        </>
                    ) : (
                        <>
                            <div className={cn(
                                "p-3 rounded-full bg-primary-500/20 text-primary-400 transition-transform duration-300 group-hover:scale-110 group-hover:rotate-3",
                                isDragging ? "animate-bounce" : ""
                            )}>
                                <Upload size={32} />
                            </div>
                            <div>
                                <p className="text-white font-semibold text-lg">
                                    Arrastra tu archivo aquí
                                </p>
                                <p className="text-sm text-gray-400">
                                    o haz click para buscar
                                </p>
                            </div>
                        </>
                    )}
                </div>

                {error && (
                    <div className="absolute -bottom-12 left-0 right-0 flex items-center gap-2 text-red-400 text-sm justify-center animate-in slide-in-from-top-2 fade-in">
                        <AlertCircle size={16} />
                        <span>{error}</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default FileUpload;
