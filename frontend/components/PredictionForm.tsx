'use client';

import { useState } from 'react';
import { getPrediction, PredictionInput } from '@/lib/api';

interface PredictionFormProps {
    onResult: (result: any) => void;
    onError: (error: string) => void;
}

export default function PredictionForm({ onResult, onError }: PredictionFormProps) {
    const [inputMode, setInputMode] = useState<'demo' | 'manual'>('demo');
    const [sampleId, setSampleId] = useState('0');
    const [loading, setLoading] = useState(false);

    // Manual input fields
    const [heartRate, setHeartRate] = useState('75');
    const [hrv, setHrv] = useState('45');
    const [timeSinceMeal, setTimeSinceMeal] = useState('2');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        onError('');

        try {
            const input: PredictionInput = inputMode === 'demo'
                ? { sample_id: parseInt(sampleId) }
                : {
                    features: [
                        parseFloat(heartRate),
                        parseFloat(hrv),
                        1.0, // ppg_amplitude
                        0.3, // ppg_pulse_width
                        0.85, // signal_quality
                        2.5, // perfusion_index
                        97, // spo2_estimate
                        36.5, // temperature
                        5, // activity_level
                        parseFloat(timeSinceMeal)
                    ]
                };

            const result = await getPrediction(input);
            onResult(result);
        } catch (err: any) {
            onError(err.message || 'Failed to get prediction');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-4">Get AI Estimate (Demo Only)</h2>

            <form onSubmit={handleSubmit} className="space-y-4">
                {/* Input Mode Selection */}
                <div>
                    <label className="block text-sm font-medium mb-2">Input Mode:</label>
                    <div className="flex gap-4">
                        <label className="flex items-center">
                            <input
                                type="radio"
                                value="demo"
                                checked={inputMode === 'demo'}
                                onChange={(e) => setInputMode('demo')}
                                className="mr-2"
                            />
                            Use Demo Sample
                        </label>
                        <label className="flex items-center">
                            <input
                                type="radio"
                                value="manual"
                                checked={inputMode === 'manual'}
                                onChange={(e) => setInputMode('manual')}
                                className="mr-2"
                            />
                            Manual Input
                        </label>
                    </div>
                </div>

                {/* Demo Sample Selection */}
                {inputMode === 'demo' && (
                    <div>
                        <label className="block text-sm font-medium mb-2">
                            Demo Sample ID (0-99):
                        </label>
                        <select
                            value={sampleId}
                            onChange={(e) => setSampleId(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded"
                        >
                            {Array.from({ length: 100 }, (_, i) => (
                                <option key={i} value={i}>Sample {i}</option>
                            ))}
                        </select>
                    </div>
                )}

                {/* Manual Input Fields */}
                {inputMode === 'manual' && (
                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium mb-1">
                                Heart Rate (bpm):
                            </label>
                            <input
                                type="number"
                                value={heartRate}
                                onChange={(e) => setHeartRate(e.target.value)}
                                className="w-full p-2 border border-gray-300 rounded"
                                min="50"
                                max="120"
                                step="1"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1">
                                Heart Rate Variability (ms):
                            </label>
                            <input
                                type="number"
                                value={hrv}
                                onChange={(e) => setHrv(e.target.value)}
                                className="w-full p-2 border border-gray-300 rounded"
                                min="20"
                                max="80"
                                step="1"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1">
                                Time Since Last Meal (hours):
                            </label>
                            <input
                                type="number"
                                value={timeSinceMeal}
                                onChange={(e) => setTimeSinceMeal(e.target.value)}
                                className="w-full p-2 border border-gray-300 rounded"
                                min="0"
                                max="8"
                                step="0.5"
                            />
                        </div>
                        <p className="text-xs text-gray-500">
                            Note: Other features use default values for this demo
                        </p>
                    </div>
                )}

                {/* Submit Button */}
                <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
                >
                    {loading ? 'Processing...' : 'Get AI Estimate (Demo Only)'}
                </button>
            </form>
        </div>
    );
}
