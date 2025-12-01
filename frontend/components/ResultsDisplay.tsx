import { PredictionOutput } from '@/lib/api';

interface ResultsDisplayProps {
    result: PredictionOutput | null;
    error: string;
}

export default function ResultsDisplay({ result, error }: ResultsDisplayProps) {
    if (error) {
        return (
            <div className="bg-red-100 border border-red-400 text-red-700 p-4 rounded-lg">
                <h3 className="font-bold mb-2">Error</h3>
                <p>{error}</p>
            </div>
        );
    }

    if (!result) {
        return null;
    }

    const getRangeColor = (range: string) => {
        switch (range) {
            case 'low':
                return 'bg-yellow-100 border-yellow-500 text-yellow-800';
            case 'normal':
                return 'bg-green-100 border-green-500 text-green-800';
            case 'high':
                return 'bg-red-100 border-red-500 text-red-800';
            default:
                return 'bg-gray-100 border-gray-500 text-gray-800';
        }
    };

    const getRangeBadgeColor = (range: string) => {
        switch (range) {
            case 'low':
                return 'bg-yellow-500';
            case 'normal':
                return 'bg-green-500';
            case 'high':
                return 'bg-red-500';
            default:
                return 'bg-gray-500';
        }
    };

    return (
        <div className="space-y-4">
            <div className={`border-2 p-6 rounded-lg ${getRangeColor(result.glucose_range)}`}>
                <h3 className="text-xl font-bold mb-4">AI Estimate Results (Demo)</h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div>
                        <p className="text-sm font-medium mb-1">Estimated Glucose:</p>
                        <p className="text-3xl font-bold">{result.estimated_glucose} mg/dL</p>
                    </div>

                    <div>
                        <p className="text-sm font-medium mb-1">Range:</p>
                        <span className={`inline-block px-4 py-2 rounded-full text-white font-bold text-lg ${getRangeBadgeColor(result.glucose_range)}`}>
                            {result.glucose_range.toUpperCase()}
                        </span>
                    </div>
                </div>

                <div className="mb-4">
                    <p className="text-sm font-medium mb-1">Confidence:</p>
                    <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-4">
                            <div
                                className="bg-blue-600 h-4 rounded-full"
                                style={{ width: `${result.confidence * 100}%` }}
                            />
                        </div>
                        <span className="font-bold">{(result.confidence * 100).toFixed(0)}%</span>
                    </div>
                </div>

                <div className="text-xs">
                    <p className="font-medium">Model Used: {result.model_used}</p>
                </div>
            </div>

            {/* Disclaimer in results */}
            <div className="bg-yellow-50 border-2 border-yellow-400 p-4 rounded-lg">
                <p className="text-sm font-bold text-yellow-800 mb-2">⚠️ IMPORTANT REMINDER:</p>
                <p className="text-xs text-yellow-700">
                    This is a research demo using synthetic data. NOT for medical use.
                    Always consult a healthcare professional for actual glucose monitoring and medical advice.
                </p>
            </div>
        </div>
    );
}
