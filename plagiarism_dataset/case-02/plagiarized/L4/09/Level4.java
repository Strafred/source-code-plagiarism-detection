import java.util.Scanner;
public class Level4 {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder: ");
        double radius = input.nextDouble();
        double tinggi = input.nextDouble();
        System.out.println("The area is " + hitungLuasAlas(radius));
        System.out.println("The volume of the cylinder is " + hitungVolume(hitungLuasAlas(radius), tinggi));
    }
    public static double hitungLuasAlas(double rad) {
        double luasAlas = rad * rad * 3.14159;
        return luasAlas;
    }
    public static double hitungVolume(double luasAlas, double tinggi) {
        double volume = luasAlas * tinggi;
        return volume;
    }
}
