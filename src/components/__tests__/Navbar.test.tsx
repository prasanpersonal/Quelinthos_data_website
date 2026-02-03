import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Navbar from '../Navbar.tsx';

describe('Navbar', () => {
  it('renders all 3 nav items', () => {
    render(<Navbar onNavigate={() => {}} />);
    expect(screen.getByText('Home')).toBeInTheDocument();
    expect(screen.getByText('Solutions')).toBeInTheDocument();
    expect(screen.getByText('Contact')).toBeInTheDocument();
  });

  it('calls onNavigate with correct section ID on click', () => {
    const onNavigate = vi.fn();
    render(<Navbar onNavigate={onNavigate} />);
    fireEvent.click(screen.getByText('Solutions'));
    expect(onNavigate).toHaveBeenCalledWith('portfolio');
  });

  it('shows status label "OPEN FOR PROJECTS"', () => {
    render(<Navbar onNavigate={() => {}} />);
    expect(screen.getByText('OPEN FOR PROJECTS')).toBeInTheDocument();
  });
});
