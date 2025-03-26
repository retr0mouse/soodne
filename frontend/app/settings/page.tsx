'use client'

import { useState } from 'react';

export default function Settings() {
    // Add state for checkboxes
    const [selectedStores, setSelectedStores] = useState({
        rimi: true,
        coop: true,
        maxima: true,
        prisma: true
    });

    // Handle checkbox changes
    const handleCheckboxChange = (store: keyof typeof selectedStores) => {
        setSelectedStores(prev => ({
            ...prev,
            [store]: !prev[store]
        }));
    };

    function save() {
        localStorage.setItem('stores', JSON.stringify(selectedStores));
    }

    return (
        <main className="container mx-auto py-8">
            <h1 className="text-4xl font-bold mb-6 place-self-center">Settings</h1>
            <h2 className="text-2xl mb-3">Show only selected stores:</h2>
            <div className="form-control">
                <label className="label cursor-pointer">
                    <span className="label-text text-2xl font-sans">Rimi</span>
                    <input 
                        type="checkbox" 
                        checked={selectedStores.rimi}
                        onChange={() => handleCheckboxChange('rimi')}
                        className="checkbox checkbox-primary"
                    />
                </label>
                <label className="label cursor-pointer">
                    <span className="label-text text-2xl font-sans">Coop</span>
                    <input 
                        type="checkbox" 
                        checked={selectedStores.coop}
                        onChange={() => handleCheckboxChange('coop')}
                        className="checkbox checkbox-primary"
                    />
                </label>
                <label className="label cursor-pointer">
                    <span className="label-text text-2xl font-sans">Maxima</span>
                    <input 
                        type="checkbox" 
                        checked={selectedStores.maxima}
                        onChange={() => handleCheckboxChange('maxima')}
                        className="checkbox checkbox-primary"
                    />
                </label>
                <label className="label cursor-pointer">
                    <span className="label-text text-2xl font-sans">Prisma</span>
                    <input 
                        type="checkbox" 
                        checked={selectedStores.prisma}
                        onChange={() => handleCheckboxChange('prisma')}
                        className="checkbox checkbox-primary"
                    />
                </label>
            </div>
            <button 
                className="btn btn-primary mt-4"
                onClick={save}
            >
                Save
            </button>
        </main>
    );
}
