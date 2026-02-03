import { forwardRef, type ReactNode } from 'react';

function createMotionComponent(tag: string) {
  return forwardRef<HTMLElement, Record<string, unknown>>((props, ref) => {
    const {
      initial: _initial,
      animate: _animate,
      exit: _exit,
      transition: _transition,
      variants: _variants,
      whileHover: _whileHover,
      whileTap: _whileTap,
      whileInView: _whileInView,
      viewport: _viewport,
      layout: _layout,
      layoutId: _layoutId,
      ...rest
    } = props;
    const Tag = tag as 'div';
    return <Tag ref={ref as React.Ref<HTMLDivElement>} {...rest} />;
  });
}

export const motion = {
  div: createMotionComponent('div'),
  section: createMotionComponent('section'),
  button: createMotionComponent('button'),
  nav: createMotionComponent('nav'),
  span: createMotionComponent('span'),
  p: createMotionComponent('p'),
  h1: createMotionComponent('h1'),
  h2: createMotionComponent('h2'),
  a: createMotionComponent('a'),
};

export function AnimatePresence({ children }: { children?: ReactNode }) {
  return <>{children}</>;
}

export function useInView() {
  return true;
}

const mockMotionValue = {
  get: () => 0,
  set: () => {},
  onChange: () => () => {},
};

export function useScroll() {
  return { scrollYProgress: mockMotionValue, scrollXProgress: mockMotionValue };
}

export function useTransform() {
  return mockMotionValue;
}

export function useSpring() {
  return mockMotionValue;
}
