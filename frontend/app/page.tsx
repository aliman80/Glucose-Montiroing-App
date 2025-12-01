'use client';

import { useState } from 'react';
import DisclaimerBanner from '@/components/DisclaimerBanner';
import PredictionForm from '@/components/PredictionForm';
import ResultsDisplay from '@/components/ResultsDisplay';
import { PredictionOutput } from '@/lib/api';

export default function Home() {
    const [result, setResult] = useState<PredictionOutput | null>(null);
    const [error, setError] = useState('');

    return (
        <main className="min-h-screen p-8 bg-gradient-to-b from-gray-50 to-gray-100">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold mb-2">
                        Non-Invasive Glucose Monitor
                    </h1>
                    <p className="text-xl text-gray-600">
                        Research Demonstration Only
                    </p>
                </div>

                {/* Disclaimer Banner */}
                <DisclaimerBanner />

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Left: Input Form */}
                    <div>
                        <PredictionForm
                            onResult={(r) => {
                                setResult(r);
                                setError('');
                            }}
                            onError={(e) => {
                                setError(e);
                                setResult(null);
                            }}
                        />
                    </div>

                    {/* Right: Results */}
                    <div>
                        <ResultsDisplay result={result} error={error} />
                    </div>
                </div>

                {/* Footer */}
                <footer className="mt-12 text-center text-sm text-gray-500">
                    <p className="mb-2">
                        <strong>Research Demo Only</strong> • Not a Medical Device • Not FDA Approved
                    </p>
                    <p>
                        This system uses synthetic data for educational purposes.
                        Always consult a qualified healthcare professional for medical advice.
                    </p>
                </footer>
            </div>
        </main>
    );
}
