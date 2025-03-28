import java.util.Scanner;
public class Kasus2 {
    public static double hitungArea(double r) {
        return r * r * Math.PI;
    }
    public static double hitungVolume(double area, double l) {
        return area * l;
    }
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder: ");
        double r = input.nextDouble();
        double l = input.nextDouble();
        double area = hitungArea(r);
        double volume = hitungVolume(area, l);
        System.out.println("The area is " + area);
        System.out.println("The volume of the cylinder is " + volume);
    }
}
