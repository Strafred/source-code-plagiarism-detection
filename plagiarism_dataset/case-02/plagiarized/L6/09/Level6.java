import java.util.Scanner;
public class Level6 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        double radius, tinggi;
        System.out.print("Enter the radius and length of a cylinder: ");
        radius = in.nextDouble(); tinggi = in.nextDouble();
        count(radius, tinggi);
    }
    private static void count(double rad, double height) {
        System.out.println("The area is " + Math.pow(rad, 2) * Math.PI);
        System.out.println("The volume of the cylinder is " + Math.pow(rad, 2) * Math.PI * height);
    }
}
