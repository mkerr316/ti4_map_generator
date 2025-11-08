import React from "react";
import { Button, Form, Alert } from "react-bootstrap";
import evaluators from "../balance/default_evaluators.json";

/**
 * BalanceOptions component provides UI controls for map balancing
 * Uses the balance algorithm from ti4-map-lab to improve map fairness
 */
class BalanceOptions extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            selectedEvaluator: 0, // Default to "Simple Slice"
            balanceGap: null,
            isBalancing: false,
            balanceIterations: 100,
        };

        this.handleEvaluatorChange = this.handleEvaluatorChange.bind(this);
        this.handleIterationsChange = this.handleIterationsChange.bind(this);
        this.handleBalanceClick = this.handleBalanceClick.bind(this);
    }

    handleEvaluatorChange(event) {
        this.setState({ selectedEvaluator: parseInt(event.target.value) });
    }

    handleIterationsChange(event) {
        const value = parseInt(event.target.value);
        if (value > 0 && value <= 1000) {
            this.setState({ balanceIterations: value });
        }
    }

    handleBalanceClick() {
        if (this.props.onBalanceClick) {
            this.setState({ isBalancing: true });
            const evaluator = evaluators[this.state.selectedEvaluator];

            // Call the parent's balance function
            this.props.onBalanceClick(
                evaluator,
                this.state.balanceIterations,
                (balanceGap) => {
                    this.setState({
                        balanceGap: balanceGap,
                        isBalancing: false
                    });
                }
            );
        }
    }

    render() {
        const currentEvaluator = evaluators[this.state.selectedEvaluator];

        return (
            <div className="balance-options">
                <h5>Map Balance</h5>
                <p className="text-muted small">
                    Improve map balance by swapping systems to reduce advantage gaps between players.
                </p>

                <Form.Group className="mb-3">
                    <Form.Label>Balance Strategy</Form.Label>
                    <Form.Select
                        value={this.state.selectedEvaluator}
                        onChange={this.handleEvaluatorChange}
                    >
                        {evaluators.map((evaluator, index) => (
                            <option key={index} value={index}>
                                {evaluator.title}
                            </option>
                        ))}
                    </Form.Select>
                    <Form.Text className="text-muted">
                        {this.getEvaluatorDescription(currentEvaluator.title)}
                    </Form.Text>
                </Form.Group>

                <Form.Group className="mb-3">
                    <Form.Label>Balance Iterations</Form.Label>
                    <Form.Control
                        type="number"
                        min="1"
                        max="1000"
                        value={this.state.balanceIterations}
                        onChange={this.handleIterationsChange}
                    />
                    <Form.Text className="text-muted">
                        Higher values = better balance but slower. Recommended: 100-200.
                    </Form.Text>
                </Form.Group>

                {this.state.balanceGap !== null && (
                    <Alert variant={this.state.balanceGap < 5 ? "success" : this.state.balanceGap < 10 ? "warning" : "danger"}>
                        <strong>Balance Gap:</strong> {this.state.balanceGap.toFixed(2)}
                        <br />
                        <small>
                            {this.state.balanceGap < 5 && "Excellent balance!"}
                            {this.state.balanceGap >= 5 && this.state.balanceGap < 10 && "Good balance. Consider more iterations for better results."}
                            {this.state.balanceGap >= 10 && "Map could be more balanced. Try running balance again."}
                        </small>
                    </Alert>
                )}

                <div className="d-grid gap-2">
                    <Button
                        variant="primary"
                        onClick={this.handleBalanceClick}
                        disabled={this.state.isBalancing || !this.props.mapGenerated}
                    >
                        {this.state.isBalancing ? "Balancing..." : "Improve Balance"}
                    </Button>
                </div>

                {!this.props.mapGenerated && (
                    <Alert variant="info" className="mt-3">
                        Generate a map first before running the balance algorithm.
                    </Alert>
                )}
            </div>
        );
    }

    getEvaluatorDescription(title) {
        const descriptions = {
            "Simple Slice": "Basic slice evaluation focusing on total resources and influence within 2 spaces.",
            "Joebrew": "Advanced evaluation with distance weighting, tech bonuses, and special system modifiers.",
            "Resource Access": "Focuses purely on resource accessibility for economic advantage.",
            "Influence Access": "Focuses purely on influence accessibility for political advantage.",
            "Tech Access": "Focuses purely on technology specialty accessibility for tech advancement."
        };
        return descriptions[title] || "Custom evaluation strategy.";
    }
}

export default BalanceOptions;
