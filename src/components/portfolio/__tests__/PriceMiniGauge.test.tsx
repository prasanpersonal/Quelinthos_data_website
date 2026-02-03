import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import PriceMiniGauge from '../PriceMiniGauge.tsx';
import type { PriceFramework } from '../../../data/types.ts';

const mockPrice: PriceFramework = {
  present: { title: 'Present', description: 'Desc', bullets: ['b1'], severity: 'critical' },
  root: { title: 'Root', description: 'Desc', bullets: ['b1'], severity: 'high' },
  impact: { title: 'Impact', description: 'Desc', bullets: ['b1'], severity: 'medium' },
  cost: { title: 'Cost', description: 'Desc', bullets: ['b1'], severity: 'low' },
  expectedReturn: { title: 'Expected', description: 'Desc', bullets: ['b1'], severity: 'high' },
};

describe('PriceMiniGauge', () => {
  it('renders 5 circles with letters P, R, I, C, E', () => {
    render(<PriceMiniGauge price={mockPrice} />);
    expect(screen.getByText('P')).toBeInTheDocument();
    expect(screen.getByText('R')).toBeInTheDocument();
    expect(screen.getByText('I')).toBeInTheDocument();
    expect(screen.getByText('C')).toBeInTheDocument();
    expect(screen.getByText('E')).toBeInTheDocument();
  });

  it('applies correct severity color classes', () => {
    const { container } = render(<PriceMiniGauge price={mockPrice} />);
    const circles = container.querySelectorAll('.rounded-full');
    expect(circles[0]).toHaveClass('bg-red-500'); // critical
    expect(circles[1]).toHaveClass('bg-orange-400'); // high
    expect(circles[2]).toHaveClass('bg-yellow-400'); // medium
    expect(circles[3]).toHaveClass('bg-green-400'); // low
    expect(circles[4]).toHaveClass('bg-orange-400'); // high
  });

  it('shows tooltip text on hover', () => {
    render(<PriceMiniGauge price={mockPrice} />);
    const pCircle = screen.getByText('P').parentElement!;
    fireEvent.mouseEnter(pCircle);
    expect(screen.getByText('Present Situation')).toBeInTheDocument();
  });
});
